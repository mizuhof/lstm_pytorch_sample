import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm1 = nn.LSTMCell(2, 256)
        self.lstm2 = nn.LSTMCell(256, 256)
        self.linear = nn.Linear(256, 2)

    def forward(self, inputs, future=0):
        """
        inputs: (B, T, 2)
        """
        h_t = torch.zeros(inputs.size(0), 256, dtype=torch.float)
        c_t = torch.zeros(inputs.size(0), 256, dtype=torch.float)
        h_t2 = torch.zeros(inputs.size(0), 256, dtype=torch.float)
        c_t2 = torch.zeros(inputs.size(0), 256, dtype=torch.float)

        inputs = inputs.permute(1, 0, 2)
        outputs = [torch.zeros_like(inputs[0])]  # 処理の都合上のダミーを入れておく
        for i, input_t in enumerate(inputs):
            lstm_input_t = input_t.clone()
            lstm_input_t[input_t == 0] = outputs[i][input_t == 0]

            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs[1:], 1).squeeze(2)

        return outputs
