from flask import Flask, request, jsonify
import random

app = Flask(__name__)
@app.route("/")
def home():
    return "<h1>Welcome to the PNR Status Checker</h1>"

# Mock PNR database
pnr_data = {
    "1234567890": {"train": "Rajdhani Express", "status": "Confirmed", "seat": "B2-34"},
    "9876543210": {"train": "Shatabdi Express", "status": "Waitlist", "seat": "N/A"},
    "5678901234": {"train": "Duronto Express", "status": "RAC", "seat": "A1-12"}
}

otp_storage = {}

@app.route("/check-pnr", methods=["POST"])
def check_pnr():
    data = request.get_json()
    pnr = data.get("pnr")
    phone = data.get("phone")

    if pnr not in pnr_data:
        return jsonify({"error": "PNR not found"}), 404

    otp = random.randint(100000, 999999)
    otp_storage[phone] = otp

    return jsonify({
        "pnr_status": pnr_data[pnr],
        "otp_sent": True
    })

@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    data = request.get_json()
    phone = data.get("phone")
    user_otp = data.get("otp")

    if phone in otp_storage and str(user_otp) == str(otp_storage[phone]):
        del otp_storage[phone]
        return jsonify({"message": "OTP verified successfully", "status": "confirmed"})
    else:
        return jsonify({"error": "Invalid OTP"}), 400

if __name__ == "__main__":
    app.run(debug=True)