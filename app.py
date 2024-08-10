from flask import Flask, request, jsonify

app = Flask(__name__)

# In-memory storage for the current mood
current_mood = {"mood": None}

@app.route("/")
def start():
    return "Server is running"

@app.route("/mood", methods=["POST"])
def post_mood():
    # Extract mood from the request
    data = request.json
    mood = data.get("mood")

    if not mood:
        return jsonify({"error": "No mood provided"}), 400

    # Update the current mood
    current_mood["mood"] = mood
    return jsonify({"message": "Mood updated successfully"}), 200

@app.route("/mood", methods=["GET"])
def get_mood():
    # Fetch the current mood
    mood = current_mood["mood"]
    if mood is None:
        return jsonify({"mood": "No mood available"}), 200

    return jsonify({"mood": mood}), 200

