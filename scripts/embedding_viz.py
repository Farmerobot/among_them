#!/usr/bin/env python3
"""Generate interactive embedding visualization for SFT dataset."""
import argparse
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import alphashape
from shapely.geometry import Polygon
from shapely.ops import unary_union
import re

def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def main(input_csv, output_html, n_clusters=None, proj='umap', regenerate=False):
    # Load data
    df = pd.read_csv(input_csv)
    texts = (df['prompt'].fillna('') + ' ' + df['model_cot_and_cleaned_output'].fillna('')).tolist()

    # Setup embedding model and paths
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # model_st = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", device=device, trust_remote_code=True)
    model_st = SentenceTransformer("all-MiniLM-L6-v2", device=device, trust_remote_code=True)
    batch_size = 1
    out_path = Path(output_html)
    emb_file = out_path.parent / f"{out_path.stem}_embeddings.npy"
    if regenerate or not emb_file.exists():
        embeddings = model_st.encode(texts, batch_size=batch_size,
                                     show_progress_bar=True, convert_to_numpy=True,
                                     normalize_embeddings=False)
        np.save(emb_file, embeddings)
    else:
        embeddings = np.load(emb_file)

    # Clustering
    group_names = df['json_file_name'].astype(str).unique()
    k = int(n_clusters) if n_clusters else len(group_names)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    df['cluster'] = clusters

    # Embedding projection (UMAP or t-SNE)
    if proj == 'umap':
        reducer = UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
    else:  # tsne
        tsne = TSNE(n_components=2, init='random', random_state=42)
        embedding_2d = tsne.fit_transform(embeddings)
    df['emb_x'], df['emb_y'] = embedding_2d[:,0], embedding_2d[:,1]

    # Plot
    fig = px.scatter(df, x='emb_x', y='emb_y', color=df['cluster'].astype(str),
                     hover_data=['json_file_name','player_name','player_role'])

    # Add alpha-shape regions for each json_file_name & player_role
    colors = px.colors.qualitative.Light24
    group_pairs = df[['json_file_name','player_role']].drop_duplicates().values.tolist()
    for i, (g, role) in enumerate(group_pairs):
        sub = df[(df['json_file_name']==g)&(df['player_role']==role)]
        if len(sub) < 4:
            continue
        # 2D points array for region
        pts = sub[['emb_x','emb_y']].values
        # compute alpha shape (concave hull). alpha parameter can be tuned
        alpha = alphashape.optimizealpha(pts)
        shape = alphashape.alphashape(pts, alpha)
        # ensure shape is a Polygon
        if isinstance(shape, Polygon):
            xs, ys = shape.exterior.xy
        else:
            # if MultiPolygon, unify boundaries
            union = unary_union(shape)
            xs, ys = union.exterior.xy
        # convert coordinate arrays to lists for Plotly
        xs, ys = list(xs), list(ys)
        fig.add_trace(go.Scatter(x=xs, y=ys,
                                 fill='toself', fillcolor=colors[i % len(colors)],
                                 line=dict(color=colors[i % len(colors)]),
                                 opacity=0.2, name=f'{g}-{role} region',
                                 hoverinfo='skip', showlegend=True))

    fig.update_layout(title=f'Embedding visualization ({proj} + KMeans)',
                      legend_title_text='Cluster')
    fig.write_html(output_html, include_plotlyjs='cdn')
    print(f"Saved visualization to {output_html}")

    # Second visualization: remove <thought> and <think> tags, then embed with tqdm
    cleaned = [re.sub(r'<thought>.*?</thought>', '', t, flags=re.DOTALL) for t in texts]
    cleaned = [re.sub(r'<think>.*?</think>', '', t, flags=re.DOTALL) for t in cleaned]
    emb_stripped_file = out_path.parent / f"{out_path.stem}_embeddings_stripped.npy"
    if regenerate or not emb_stripped_file.exists():
        embeddings2 = model_st.encode(cleaned, batch_size=batch_size,
                                      show_progress_bar=True, convert_to_numpy=True,
                                      normalize_embeddings=False)
        np.save(emb_stripped_file, embeddings2)
    else:
        embeddings2 = np.load(emb_stripped_file)
    # Clustering & projection for stripped texts
    k2 = int(n_clusters) if n_clusters else len(group_names)
    kmeans2 = KMeans(n_clusters=k2, random_state=42)
    clusters2 = kmeans2.fit_predict(embeddings2)
    if proj == 'umap':
        emb2 = UMAP(n_components=2, random_state=42).fit_transform(embeddings2)
    else:
        emb2 = TSNE(n_components=2, init='random', random_state=42).fit_transform(embeddings2)
    df2 = df.copy()
    df2['emb_x'], df2['emb_y'], df2['cluster'] = emb2[:,0], emb2[:,1], clusters2
    fig2 = px.scatter(df2, x='emb_x', y='emb_y', color=df2['cluster'].astype(str),
                      hover_data=['json_file_name','player_name','player_role'])
    for i, (g, role) in enumerate(group_pairs):
        sub2 = df2[(df2['json_file_name']==g)&(df2['player_role']==role)]
        if len(sub2) < 4:
            continue
        # 2D points array for stripped-region
        pts2 = sub2[['emb_x','emb_y']].values
        # compute alpha shape (concave hull). alpha parameter can be tuned
        alpha = alphashape.optimizealpha(pts2)
        shape = alphashape.alphashape(pts2, alpha)
        # ensure shape is a Polygon
        if isinstance(shape, Polygon):
            xs, ys = shape.exterior.xy
        else:
            # if MultiPolygon, unify boundaries
            union = unary_union(shape)
            xs, ys = union.exterior.xy
        # convert coordinate arrays to lists for Plotly
        xs, ys = list(xs), list(ys)
        fig2.add_trace(go.Scatter(x=xs, y=ys,
                                  fill='toself', fillcolor=colors[i % len(colors)],
                                  line=dict(color=colors[i % len(colors)]),
                                  opacity=0.2, name=f'{g}-{role} region',
                                  hoverinfo='skip', showlegend=True))
    fig2.update_layout(title=f'Stripped tags visualization ({proj} + KMeans)',
                       legend_title_text='Cluster')
    out_html2 = output_html.replace('.html', '_stripped.html')
    fig2.write_html(out_html2, include_plotlyjs='cdn')
    print(f"Saved stripped visualization to {out_html2}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embed & visualize SFT dataset')
    parser.add_argument('--input', default='data/sft_dataset.csv', help='CSV file path')
    parser.add_argument('--output', default='generated/embedding_viz.html', help='Output HTML file')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters for KMeans')
    parser.add_argument('--proj', choices=['umap','tsne'], default='tsne', help='Projection method (umap or tsne)')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate embeddings even if saved')
    args = parser.parse_args()
    main(args.input, args.output, args.n_clusters, args.proj, args.regenerate)
