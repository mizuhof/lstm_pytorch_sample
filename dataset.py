import math

import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, data_sample_num=1000, seq_len=10, max_missing_num=3):
        self.seq_len = seq_len
        self.data_sample_num = data_sample_num
        self.max_missing_num = max_missing_num
        self.data, self.target = self.data_load()

    def data_load(self):
        """
        return: missing_trajectory, target
            missing_trajectory: (self.data_sample_num, self.seq_len, 2)
            target:             (self.data_sample_num, self.seq_len, 2)
        """

        w = 1
        scale_x = torch.rand(self.data_sample_num, 1)
        scale_y = torch.rand(self.data_sample_num, 1)
        theta_x = torch.rand(self.data_sample_num, 1)*math.pi
        theta_y = torch.rand(self.data_sample_num, 1)*math.pi

        sequence = torch.arange(0, math.pi/2, step=math.pi / 2 / (self.seq_len+1)
                                ).repeat(self.data_sample_num).view(self.data_sample_num, -1)

        x = scale_x * torch.sin(sequence/w + theta_x)
        y = scale_y * torch.sin(sequence/w + theta_y)
        tra = torch.stack([x, y], dim=2)

        missing_tra = []
        for t in tra[:, :-1].clone():
            _from = torch.randint(8, (1,))
            _to = torch.randint(_from.item(), self.seq_len-1, (1,))
            _to = _to if (_to - _from) <= self.max_missing_num \
                else (_from + self.max_missing_num)
            dt = t
            dt[_from:_to] = 0
            missing_tra.append(dt)

        missing_tra = torch.stack(missing_tra, dim=0)

        return missing_tra, tra[:, 1:]

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.target)


if __name__ == '__main__':
    dataset = Dataset(data_sample_num=2)
    print(dataset.data)
    print(dataset.target)
    print(dataset.data.size(), dataset.target.size())
