import requests
from flask import Flask, request, jsonify

# ✅ Replace with your actual WhatsApp Cloud API credentials
ACCESS_TOKEN = "EAAJyY9crqRIBO8pUoQhJLQKuoNwL7AOpvdcPoW7JopNVrr1WbL4lC1xTLpNl2rwJkYkwUpydgtTgiqQVzjZBLlOixx30YPxUPiwfRhWKQk5eOGySyV6BEMZA1So4otBA6IoVkv43x5Nj8bNASoAt2sc3f1B3N9S1oKcfZAmxShiLTyZBPjkcTIQSFbwOqq0RRZCoLKQ4yZCaMtptDDetMWo0oh6wWnXg4ZD"
PHONE_NUMBER_ID = "652861581247652"
VERIFY_TOKEN = "FND_DEMO"  # ✅ Use the same token set in Meta Console
WHATSAPP_API_URL = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"

# ✅ Import your Fake News Detection System
from fake_news_detection_system import FakeNewsDetectionSystem  
system = FakeNewsDetectionSystem()

app = Flask(__name__)

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    """Handles WhatsApp Webhook Requests."""
    if request.method == 'GET':
        # ✅ Verify the webhook when saving it in Meta Console
        token_sent = request.args.get("hub.verify_token")
        if token_sent == VERIFY_TOKEN:
            return request.args.get("hub.challenge")
        return "Invalid verify token", 403

    elif request.method == 'POST':
        data = request.json
        if "messages" in data["entry"][0]["changes"][0]["value"]:
            message_text = data["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]
            sender_id = data["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]

            print(f"📩 Received message: {message_text} from {sender_id}")

            # ✅ Run Fake News Detection System on received claim
            results, verdict = system.analyze_claim(message_text)

            response_text = f"✅ Fake News Detection Verdict:\n{verdict}"

            # ✅ Send response back to WhatsApp user
            send_whatsapp_message(sender_id, response_text)

        return jsonify({"status": "message processed"})

def send_whatsapp_message(recipient_id, message_text):
    """Sends a WhatsApp message response."""
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_id,
        "type": "text",
        "text": {"body": message_text}
    }

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(WHATSAPP_API_URL, json=payload, headers=headers)
    print(f"📤 Sent response: {response.status_code}")

if __name__ == '__main__':
    app.run(debug=True)
