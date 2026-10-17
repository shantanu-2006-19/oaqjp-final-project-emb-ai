from flask import Flask, request, render_template_string

from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>Emotion Detection</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f7f7f7; }
        .box { max-width: 600px; margin: auto; background: white; padding: 25px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        textarea { width: 100%; height: 120px; padding: 10px; }
        .result { margin-top: 20px; padding: 15px; background: #eef8ff; border-radius: 8px; }
        .error { color: red; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="box">
        <h2>Emotion Detection</h2>
        <form method="POST">
            <textarea name="text" placeholder="Enter text to analyze">{{ request_text or '' }}</textarea><br><br>
            <button type="submit">Analyze</button>
        </form>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        {% if result %}
            <div class="result">
                <pre>{{ result }}</pre>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    request_text = ""

    if request.method == "POST":
        request_text = request.form.get("text", "")

        if not request_text.strip():
            error = "400: Blank input is not allowed. Please enter some text."
            return render_template_string(
                HTML_TEMPLATE,
                result=result,
                error=error,
                request_text=request_text
            ), 400

        try:
            result = emotion_detector(request_text)
            result = str(result)
        except ValueError as exc:
            error = str(exc)
            return render_template_string(
                HTML_TEMPLATE,
                result=result,
                error=error,
                request_text=request_text
            ), 400

    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        error=error,
        request_text=request_text
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
