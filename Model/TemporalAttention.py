import torch.nn as nn
import torch


class TemporalAttention(nn.Module):
    def __init__(self):
        super(TemporalAttention, self).__init__()
        self.dilatedConv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(8,1), dilation=(1,1), padding='same'),
            nn.BatchNorm2d(1),
            nn.ELU()
        )
        self.dilatedConv_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(8,1), dilation=(2,1), padding='same'),
            nn.BatchNorm2d(1),
            nn.ELU()
        )
        self.dilatedConv_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(8,1), dilation=(3,1), padding='same'),
            nn.BatchNorm2d(1),
            nn.ELU()
        )

        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1,1)),
            nn.BatchNorm2d(1),
            nn.ELU()
        )

        self.Wq = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(32,1), padding='same')
        self.Wk = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(32, 1), padding='same')
        self.Wv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(32, 1), padding='same')

        pass

    def forward(self, input):
        dilatedConv_1 = self.dilatedConv_1(input)
        dilatedConv_2 = self.dilatedConv_2(input)
        dilatedConv_3 = self.dilatedConv_3(input)

        concat = torch.cat([dilatedConv_1, dilatedConv_2, dilatedConv_3], 1)
        pointwiseConv = self.pointwiseConv(concat)

        Q = self.Wq(pointwiseConv).squeeze(1)
        K = self.Wk(pointwiseConv).squeeze(1)
        V = self.Wv(pointwiseConv).squeeze(1)


        QKt = torch.bmm(Q, K.transpose(1,2))
        softmax_QKt = torch.softmax(QKt, dim=1)

        output = torch.bmm(QKt, V).unsqueeze(1)
        return output
        pass
    pass
