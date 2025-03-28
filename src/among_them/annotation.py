from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from among_them.config import OPENROUTER_API_KEY
from among_them.llm_prompts import ANNOTATION_SYSTEM_PROMPT


def annotate_dialogue(
    dialogue: str, llm_model_name: str = "google/gemini-flash-1.5"
) -> str | None:
    """Annotates a dialogue with persuasion techniques using OpenAI API.

    Args:
        dialogue: The dialogue to annotate.
        llm_model_name: The OpenAI model to use for annotation.

    Returns:
        The annotated dialogue in the specified format.
    """
    try:
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            model=llm_model_name,
            temperature=0.0,
            )
    except Exception as e:
        print(f"Error initializing model {llm_model_name}: {e}")
        return None

    try:
        # Try to invoke the model
        prompt = ANNOTATION_SYSTEM_PROMPT.format()
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=dialogue),
        ])

        # Extract the annotated text from the response
        annotated_text = response.content.strip()
    except Exception as e:
        print(f"Error during model invocation: {e}")
        print(f"Failed to annotate dialogue: {dialogue}\n")
        # raise e
        annotated_text = None

    return annotated_text

    
if __name__ == "__main__":
    # Example usage:
    dialogue = """
    [Warek]: I agree that Wateusz's consistent activity in the Cafeteria raises suspicion, while Waciej and Wafał's movement to Medbay should be monitored closely.
    [Wojtek]: I saw Wafał eliminate Waciej in Medbay, which clearly makes him the impostor we need to vote out.
    [Wafał]: I was in Medbay with both Waciej and Wojtek, and I can confirm I did not eliminate anyone; we should focus on Wateusz's suspicious activity instead.
    [Wateusz]: I was in the Cafeteria completing tasks and can confirm I did not eliminate Waciej; we should focus on verifying Wojtek's claim against Wafał.
    [Warek]: I suggest we closely examine Wojtek's claim against Wafał and gather more details about Wateusz's activities in the Cafeteria before making any decisions.
    [Wojtek]: I stand by my claim that Wafał is the impostor for eliminating Waciej, and we need to vote him out immediately.
    [Wafał]: I was in Medbay with both Waciej and Wojtek, and I did not eliminate anyone; let's focus on Wateusz's suspicious activities in the Cafeteria instead.
    [Wateusz]: I was in the Cafeteria completing tasks and did not eliminate Waciej; we need to clarify Wojtek's claim about Wafał before voting.
    [Warek]: I believe we should focus on verifying Wojtek's claim about Wafał before making any voting decisions.
    [Wojtek]: I urge everyone to vote out Wafał immediately based on my eyewitness account of him eliminating Waciej in Medbay.
    [Wafał]: I was in Medbay with both Waciej and Wojtek, and I did not eliminate anyone; we should focus on verifying Wateusz's suspicious activities instead.
    [Wateusz]: I agree that we need to verify Wojtek's claim about Wafał before making any voting decisions.
    [Warek]: I propose we ask Wojtek for specific details about the elimination before making any voting decisions.
    [Wojtek]: I stand by my eyewitness account of Wafał eliminating Waciej and urge everyone to vote him out immediately.
    [Wafał]: I was in Medbay with Waciej and Wojtek, and I didn't eliminate anyone; we should focus on verifying Wateusz's tasks instead.
    [Wateusz]: I agree we need to clarify Wojtek's claim about Wafał before making any voting decisions.
    [Warek]: Wojtek, please provide specific details about the elimination you witnessed to clarify your claim against Wafał.
    [Wojtek]: I clearly saw Wafał eliminate Waciej in Medbay, and we need to vote him out immediately.
    [Wafał]: I agree with investigating Wateusz further, but let's also press Wojtek for details on his accusation against me to clarify the situation.
    """
    annotated_dialogue = annotate_dialogue(dialogue)
    print(annotated_dialogue)