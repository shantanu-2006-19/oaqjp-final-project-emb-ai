import unittest

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    def test_happy_text(self):
        result = emotion_detector("I am very happy and excited today")
        self.assertIn("anger", result)
        self.assertIn("disgust", result)
        self.assertIn("fear", result)
        self.assertIn("joy", result)
        self.assertIn("sadness", result)
        self.assertIn("dominant_emotion", result)
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_angry_text(self):
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_blank_input(self):
        with self.assertRaises(ValueError):
            emotion_detector("   ")

    def test_fear_text(self):
        result = emotion_detector("I feel scared and afraid")
        self.assertEqual(result["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()
