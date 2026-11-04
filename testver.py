
# ==========================================
# MINI AI CHATBOT VERSION 2
# PURE PYTHON
# NO IMPORTS
# NO LIBRARIES
# NO CLASSES
# NO OOP
# NO F-STRINGS
# ==========================================


# ==========================================
# CONVERSATION TRAINING DATA
# ==========================================

conversations = [

    ("hi", "hello nice to meet you"),

    ("hello", "hello how are you"),

    ("hey", "hey how are you doing"),

    ("how are you", "i am doing good thanks"),

    ("how are you doing", "i am doing great"),

    ("who are you", "i am a chatbot"),

    ("what are you", "i am an artificial intelligence chatbot"),

    ("what is your name", "my name is python bot"),

    ("your name", "my name is python bot"),

    ("what can you do", "i can talk and answer simple questions"),

    ("can you help me", "yes i can help you"),

    ("help me", "sure what do you need help with"),

    ("are you intelligent", "i am learning to understand language"),

    ("do you learn", "yes i learn patterns from training examples"),

    ("what is ai", "ai means artificial intelligence"),

    ("what is a neural network", "a neural network learns patterns using weights and biases"),

    ("how do you work", "i process words and predict responses using a neural network"),

    ("are you a robot", "i am a computer program"),

    ("are you real", "i am software running on a computer"),

    ("tell me something", "learning is the process of improving through experience"),

    ("what is python", "python is a programming language"),

    ("do you like talking", "yes i enjoy our conversation"),

    ("i am happy", "that is wonderful why are you happy"),

    ("i am sad", "i am sorry to hear that do you want to talk"),

    ("i am bored", "we can talk about something interesting"),

    ("thank you", "you are welcome"),

    ("thanks", "you are welcome"),

    ("good morning", "good morning how are you"),

    ("good night", "good night sleep well"),

    ("goodbye", "goodbye talk to you later"),

    ("bye", "goodbye have a nice day"),

    ("how old are you", "i do not have a human age"),

    ("where are you", "i exist inside this computer program"),

    ("who made you", "i was created using python code"),

    ("what do you think", "i use patterns to generate my answers"),

    ("hi who are you", "hello i am a python chatbot"),

    ("hi how are you", "hello i am doing good"),

    ("hello who are you", "hello i am a chatbot"),

    ("who are you and what can you do", "i am a chatbot and i can answer simple questions"),

    ("how does ai work", "ai learns patterns from data and uses them to make predictions"),

    ("what is machine learning", "machine learning teaches computers to learn patterns from examples")

]


# ==========================================
# BUILD VOCABULARY
# ==========================================

words = ["<PAD>", "<USER>", "<BOT>", "<END>"]

for user_text, bot_text in conversations:

    for word in user_text.lower().split():

        if word not in words:
            words.append(word)

    for word in bot_text.lower().split():

        if word not in words:
            words.append(word)


vocab_size = len(words)


def get_id(word):

    if word in words:
        return words.index(word)

    return -1


PAD = get_id("<PAD>")
USER = get_id("<USER>")
BOT = get_id("<BOT>")
END = get_id("<END>")


# ==========================================
# NEURAL NETWORK SETTINGS
# ==========================================

context_size = 8
hidden_size = 32
learning_rate = 0.035

input_size = context_size * vocab_size
output_size = vocab_size


# ==========================================
# INITIALIZE WEIGHTS
# ==========================================

W1 = []

for i in range(hidden_size):

    row = []

    for j in range(input_size):

        value = (((i + 1) * (j + 3)) % 17 - 8) * 0.01

        row.append(value)

    W1.append(row)


B1 = [0.0 for i in range(hidden_size)]


W2 = []

for i in range(output_size):

    row = []

    for j in range(hidden_size):

        value = (((i + 2) * (j + 5)) % 19 - 9) * 0.01

        row.append(value)

    W2.append(row)


B2 = [0.0 for i in range(output_size)]


# ==========================================
# ACTIVATION FUNCTIONS
# ==========================================

def relu(x):

    if x > 0:
        return x

    return 0.0


def relu_derivative(x):

    if x > 0:
        return 1.0

    return 0.0


def softmax(values):

    largest = max(values)

    result = []

    for value in values:

        exponent = 2.718281828 ** (value - largest)

        result.append(exponent)

    total = sum(result)

    for i in range(len(result)):

        result[i] = result[i] / total

    return result


# ==========================================
# PREPARE CONTEXT
# ==========================================

def make_context(sequence):

    context = [PAD for i in range(context_size)]

    recent = sequence[-context_size:]

    start = context_size - len(recent)

    for i in range(len(recent)):

        context[start + i] = recent[i]

    return context


# ==========================================
# FORWARD PROPAGATION
# ==========================================

def forward(sequence):

    context = make_context(sequence)

    hidden = []
    raw_hidden = []

    # Only one input is active per context position.
    # This avoids calculating every zero input.

    for i in range(hidden_size):

        value = B1[i]

        for position in range(context_size):

            word = context[position]

            index = position * vocab_size + word

            value += W1[i][index]

        raw_hidden.append(value)

        hidden.append(relu(value))

    output = []

    for i in range(output_size):

        value = B2[i]

        for j in range(hidden_size):

            value += W2[i][j] * hidden[j]

        output.append(value)

    probabilities = softmax(output)

    return context, hidden, raw_hidden, probabilities


# ==========================================
# BACKPROPAGATION
# ==========================================

def train(sequence, target):

    context, hidden, raw_hidden, probabilities = forward(sequence)

    # Output error

    output_error = probabilities[:]

    output_error[target] -= 1.0

    # Hidden layer error

    hidden_error = [0.0 for i in range(hidden_size)]

    for j in range(hidden_size):

        for i in range(output_size):

            hidden_error[j] += output_error[i] * W2[i][j]

        hidden_error[j] *= relu_derivative(raw_hidden[j])

    # Update output weights

    for i in range(output_size):

        for j in range(hidden_size):

            W2[i][j] -= (
                learning_rate *
                output_error[i] *
                hidden[j]
            )

        B2[i] -= learning_rate * output_error[i]

    # Update input weights

    for j in range(hidden_size):

        for position in range(context_size):

            word = context[position]

            index = position * vocab_size + word

            W1[j][index] -= (
                learning_rate *
                hidden_error[j]
            )

        B1[j] -= learning_rate * hidden_error[j]


# ==========================================
# CREATE TRAINING SEQUENCES
# ==========================================

training_data = []

for user_text, bot_text in conversations:

    sequence = [USER]

    for word in user_text.lower().split():

        sequence.append(get_id(word))

    sequence.append(BOT)

    response = bot_text.lower().split()

    for word in response:

        training_data.append(
            (sequence[:], get_id(word))
        )

        sequence.append(get_id(word))

    training_data.append((sequence[:], END))


# ==========================================
# TRAIN THE NETWORK
# ==========================================

print("Training neural network...")

epochs = 400

for epoch in range(epochs):

    for sequence, target in training_data:

        train(sequence, target)

print("Training finished!")

print("Vocabulary size: " + str(vocab_size))
print("Hidden neurons: " + str(hidden_size))
print("Training examples: " + str(len(training_data)))


# ==========================================
# GENERATE RESPONSE
# ==========================================

def generate(sequence, maximum_words):

    response = []

    for step in range(maximum_words):

        context, hidden, raw, probabilities = forward(sequence)

        # Find highest probability token

        best = 0

        for i in range(1, output_size):

            if probabilities[i] > probabilities[best]:

                best = i

        # Stop generation

        if best == END:

            break

        if best == PAD or best == USER or best == BOT:

            break

        response.append(words[best])

        sequence.append(best)

    return " ".join(response)


# ==========================================
# TEXT PROCESSING
# ==========================================

def clean_text(text):

    result = ""

    for character in text.lower():

        if character.isalnum() or character == " ":

            result += character

        else:

            result += " "

    return result.split()


# ==========================================
# CHAT MEMORY
# ==========================================

history = []


# ==========================================
# CHAT LOOP
# ==========================================

print("")
print("================================")
print("       PYTHON AI CHATBOT")
print("================================")
print("Type exit to stop.")
print("")

while True:

    user_input = input("YOU: ").strip()

    if user_input.lower() == "exit":

        print("BOT: Goodbye!")

        break

    user_words = clean_text(user_input)

    if len(user_words) == 0:

        print("BOT: Please say something.")

        continue

    # Start current conversation turn

    sequence = history[:]

    sequence.append(USER)

    unknown_words = 0

    for word in user_words:

        word_number = get_id(word)

        if word_number == -1:

            unknown_words += 1

        else:

            sequence.append(word_number)

    if len(sequence) == 0:

        print("BOT: I don't understand.")

        continue

    # Tell network to generate a bot response

    sequence.append(BOT)

    answer = generate(sequence, 15)

    if answer == "":

        answer = "i am still learning"

    print("BOT: " + answer)

    # Save conversation context

    history.append(USER)

    for word in user_words:

        word_number = get_id(word)

        if word_number != -1:

            history.append(word_number)

    history.append(BOT)

    for word in answer.split():

        word_number = get_id(word)

        if word_number != -1:

            history.append(word_number)

    # Keep history manageable

    if len(history) > 30:

        history = history[-30:]
