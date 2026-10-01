import numpy as np
import re
import random

class UltimateEngineJamesBot:
    def __init__(self):
        self.version = "4.0 (Neural AI Engine)"
        
        # 1. Define a local vocab sheet to map incoming strings to numerical vectors
        self.vocabulary = [
            "hey", "hi", "hello", "yo", "sup", "howdy",
            "code", "coding", "python", "ai", "computer", "software", "program", "bot", "script",
            "game", "games", "play", "playing", "xbox", "playstation", "minecraft", "roblox", "pc",
            "school", "math", "science", "class", "subject", "study", "teacher", "homework"
        ]
        self.vocab_size = len(self.vocabulary)
        self.num_classes = 4  # 0: Greetings, 1: Tech, 2: Gaming, 3: School
        self.class_names = ["greetings", "tech", "gaming", "school"]
        self.current_mood_topic = "greetings"
        self.turns_learned = 0

        # 2. Initialize Layer Weights and Biases from scratch using random numbers
        # Input Layer -> Hidden Layer (8 neurons) -> Output Layer (4 neurons)
        self.hidden_size = 8
        self.W1 = np.random.randn(self.vocab_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.num_classes) * 0.1
        self.b2 = np.zeros((1, self.num_classes))

        # Static conversational response nodes
        self.responses = {
            "greetings": ["Hey what is up! Just running some local neural activation loops over here.", "Hello again! My weights are perfectly optimized to speak with you today.", "Yo! No internet connection required here!"],
            "tech": ["Coding is brilliant once you realize it is all just matrix multiplication under the hood.", "Python scripts running deep learning nets from scratch hit completely different.", "The matrix is the foundation of all computation."],
            "gaming": ["Video games are pure masterclasses in logic routing and physics simulations.", "Minecraft is incredible because you can construct fully active redstone networks.", "Multiplayer gaming teaches you about distributed systems!"],
            "school": ["School is a solid proving ground, but engineering real AI parameters is way more exciting.", "Mathematics class gets infinitely better when you start applying algebra to cutting-edge neural networks.", "Knowledge compounds over time, just like learning iterations."]
        }
        
        # Diagnostic tracking values
        self.last_seed_source = "Neural Network Inference"
        self.last_tier_trace = {}
        self.context_scores = {"greetings": 0.0, "tech": 0.0, "gaming": 0.0, "school": 0.0}

        # ===== NEW: TELEMETRY TRACKING =====
        self.error_history = []
        self.gradient_history = []
        self.max_history_length = 50
        self.learning_rate = 0.2
        self.cumulative_error = 0.0
        self.last_gradient_magnitude = 0.0

    def clean_text(self, text):
        text = text.lower().replace("'", "").replace('"', "")
        return re.sub(r'[^\w\s]', ' ', text).strip()

    # Activation function to squash numbers smoothly between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    # Derivative function used by backpropagation to calculate error speed
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Turns raw english characters into an input vector layer of 1s and 0s
    def text_to_vector(self, text):
        cleaned = self.clean_text(text)
        words = cleaned.split()
        vector = np.zeros((1, self.vocab_size))
        for word in words:
            if word in self.vocabulary:
                idx = self.vocabulary.index(word)
                vector[0, idx] = 1.0
        return vector

    # FORWARD PASS: Pass values through mathematical layers to get prediction scores
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    # BACKPROPAGATION: Learn directly from errors by updating raw weights parameters
    def train_step(self, X, target_class_idx, lr=None):
        if lr is None:
            lr = self.learning_rate

        # Create output target array (One-Hot Encoding)
        y = np.zeros((1, self.num_classes))
        y[0, target_class_idx] = 1.0

        # Run forward calculation to find current model state
        output = self.forward(X)

        # Compute output layer error matrix margin
        error_output = y - output
        
        # ===== NEW: Track error magnitude =====
        error_magnitude = np.linalg.norm(error_output)
        self.error_history.append(float(error_magnitude))
        if len(self.error_history) > self.max_history_length:
            self.error_history.pop(0)
        self.cumulative_error += float(error_magnitude)
        
        d_output = error_output * self.sigmoid_derivative(output)

        # ===== NEW: Track gradient magnitude =====
        gradient_magnitude = np.linalg.norm(d_output)
        self.gradient_history.append(float(gradient_magnitude))
        if len(self.gradient_history) > self.max_history_length:
            self.gradient_history.pop(0)
        self.last_gradient_magnitude = float(gradient_magnitude)

        # Compute hidden layer error matrix margin
        error_hidden = np.dot(d_output, self.W2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        # Apply corrections to synaptic connection weights
        self.W2 += np.dot(self.a1.T, d_output) * lr
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * lr
        self.W1 += np.dot(X.T, d_hidden) * lr
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    def generate_reply(self, user_message):
        X = self.text_to_vector(user_message)
        
        # If user text matches no keywords, default to greetings rather than breaking matrix paths
        if np.sum(X) == 0:
            self.current_mood_topic = "greetings"
            self.context_scores = {k: 0.25 for k in self.class_names}
            return "Hey! Type some keywords about gaming, tech, or school so my neural weights can activate."

        # Compute output prediction probabilities using our neural pathways
        probabilities = self.forward(X)[0]
        
        # Map raw array indices back to descriptive strings
        predicted_idx = np.argmax(probabilities)
        self.current_mood_topic = self.class_names[predicted_idx]
        
        # Populate web dashboard diagnostic readouts
        for i, class_name in enumerate(self.class_names):
            self.context_scores[class_name] = float(probabilities[i])

        self.last_tier_trace = {
            "Layer_1_Neurons": self.hidden_size,
            "Active_Synapses": self.vocab_size * self.hidden_size,
            "Output_Confidence": round(float(probabilities[predicted_idx]), 4)
        }

        # Automatically train on the input to adapt his brain for the next turn
        self.train_step(X, predicted_idx)
        self.turns_learned += 1

        return random.choice(self.responses[self.current_mood_topic])

    # ===== NEW: Get telemetry data for frontend visualization =====
    def get_telemetry(self):
        """Extract current neural network state for real-time dashboard monitoring"""
        
        # Calculate average error rate (recent window)
        avg_error_rate = np.mean(self.error_history) if self.error_history else 0.0
        
        # Calculate error trend (is error increasing or decreasing?)
        if len(self.error_history) >= 2:
            error_trend = self.error_history[-1] - self.error_history[0]
        else:
            error_trend = 0.0
        
        # Get current confidence (max probability from last inference)
        current_confidence = max(self.context_scores.values()) if self.context_scores else 0.0
        
        # Count active synapses (connections with significant weights)
        active_synapses_w1 = np.sum(np.abs(self.W1) > 0.01)
        active_synapses_w2 = np.sum(np.abs(self.W2) > 0.01)
        active_synapses = int(active_synapses_w1 + active_synapses_w2)
        
        return {
            "turns_learned": self.turns_learned,
            "active_topic": self.current_mood_topic,
            "learning_rate": round(self.learning_rate, 4),
            "error_rate": round(float(avg_error_rate), 4),
            "error_trend": round(float(error_trend), 4),
            "confidence": round(float(current_confidence), 4),
            "gradient_magnitude": round(self.last_gradient_magnitude, 4),
            "active_synapses": active_synapses,
            "hidden_neurons": self.hidden_size,
            "cumulative_error": round(float(self.cumulative_error), 4),
            "weights": {k: round(v, 2) for k, v in self.context_scores.items()}
        }
