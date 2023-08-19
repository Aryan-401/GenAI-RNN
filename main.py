import random
import torch
import string
import unidecode

from RNN import RNN
import streamlit as st

all_characters = string.printable
n_characters = len(all_characters)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator():
    def __init__(self):
        self.chunk_len = 250
        self.num_epochs = 5000
        self.batch_size = 1
        self.print_every = 200
        self.hidden_size = 256
        self.num_layers = 2
        self.file_shakespeare = unidecode.unidecode(open('./shakespeare.txt').read())
        self.file = self.file_shakespeare
        self.lr = 0.003


    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters.index(string[c])
        return tensor

    def get_random_batch(self):
        start_index = random.randint(0, len(self.file) - self.chunk_len)
        end_index = start_index + self.chunk_len + 1
        text_str = self.file[start_index: end_index]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i,:] = self.char_tensor(text_str[:-1])
            text_target[i,:] = self.char_tensor(text_str[1:])

        return text_input.long(), text_target.long()

    def generate(self, initial_string = 'A', prediction_len = 100, temprature = 1.0):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_string)
        predicted = initial_string

        for p in range(len(initial_string) - 1):
            _, (hidden, cell) = self.rnn(initial_input[p].view(1).to(device), hidden, cell)

        last_char = initial_input[-1]

        for p in range(prediction_len):
            output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)
            output_dist = output.data.view(-1).div(temprature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        return predicted

    def get_model(self, model_name):
        self.rnn = RNN(n_characters, self.hidden_size, self.num_layers, n_characters).to(device)
        self.rnn.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
        self.rnn.eval()
        return self.rnn


model = Generator()
model.get_model('./shakespeare_model_4400.pt')

st.title('Shakespeare Style Text Generator')
st.markdown('<h3 style="color:gray;">Made by: <a href="https://aryann.tech">Aryan Garg</a></h3>', unsafe_allow_html=True)
length = st.slider('Max Length', 0, 1000, 100)
temp = st.slider('Temperature', 0.0, 1.0, 0.8)
text = st.text_area("Enter starting string", "Hallowed by thy name")
st.code(model.generate(text, length, temp))

