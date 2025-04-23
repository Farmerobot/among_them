#!/usr/bin/env python3
"""Generate interactive embedding visualization for SFT dataset."""
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import re
import numpy as np
from tqdm.auto import tqdm

def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def main(input_csv, output_html, n_clusters=None, proj='umap'):
    # Load data
    df = pd.read_csv(input_csv)
    texts = (df['prompt'].fillna('') + ' ' + df['model_cot_and_cleaned_output'].fillna('')).tolist()

    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
    model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(device)
    model.eval()

    # Tokenize & embed with tqdm progress
    batch_size = 1
    all_embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=8192, return_tensors='pt')
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            out = model(ids, attention_mask=mask, return_dict=True)
        tok_emb = out.last_hidden_state
        pooled = mean_pooling(tok_emb, mask)
        all_embeds.append(pooled.cpu().numpy())
    embeddings = np.concatenate(all_embeds, axis=0)

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

    # Add convex hulls for each json_file_name & player_role
    colors = px.colors.qualitative.Light24
    group_pairs = df[['json_file_name','player_role']].drop_duplicates().values.tolist()
    for i, (g, role) in enumerate(group_pairs):
        sub = df[(df['json_file_name']==g)&(df['player_role']==role)]
        if len(sub) < 3:
            continue
        pts = sub[['emb_x','emb_y']].values
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        fig.add_trace(go.Scatter(x=hull_pts[:,0], y=hull_pts[:,1],
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
    # Embed stripped texts with tqdm
    all_embeds2 = []
    for i in tqdm(range(0, len(cleaned), batch_size), desc="Embedding stripped texts"):
        batch2 = cleaned[i:i+batch_size]
        enc2 = tokenizer(batch2, padding=True, truncation=True, max_length=8192, return_tensors='pt')
        ids2 = enc2['input_ids'].to(device)
        mask2 = enc2['attention_mask'].to(device)
        with torch.no_grad():
            out2 = model(ids2, attention_mask=mask2, return_dict=True)
        tok_emb2 = out2.last_hidden_state
        pooled2 = mean_pooling(tok_emb2, mask2)
        all_embeds2.append(pooled2.cpu().numpy())
    embeddings2 = np.concatenate(all_embeds2, axis=0)
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
        if len(sub2) < 3:
            continue
        pts2 = sub2[['emb_x','emb_y']].values
        hull2 = ConvexHull(pts2)
        hull_pts2 = pts2[hull2.vertices]
        fig2.add_trace(go.Scatter(x=hull_pts2[:,0], y=hull_pts2[:,1],
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
    parser.add_argument('--input', default='data/sft_dataset2.csv', help='CSV file path')
    parser.add_argument('--output', default='data/embedding_viz.html', help='Output HTML file')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters for KMeans')
    parser.add_argument('--proj', choices=['umap','tsne'], default='tsne', help='Projection method (umap or tsne)')
    args = parser.parse_args()
    main(args.input, args.output, args.n_clusters, args.proj)
