import os
from dotenv import load_dotenv
from google import genai

# 1. Unlock the Vault
load_dotenv()

# --- DEBUG TRIPWIRE ---
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ ERROR: Python cannot find the GEMINI_API_KEY. The vault is sealed.")
else:
    print("✅ SUCCESS: Key found! Length:", len(api_key), "characters.")
# ----------------------

# 2. Start the Engine 
client = genai.Client()

print("Waking up Paper Brain...")

# 3. Send the Prompt
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Explain the beauty of applied probability in exactly one short sentence.'
)

# 4. Print the Response
print("\nPaper Brain says:")
print(response.text)