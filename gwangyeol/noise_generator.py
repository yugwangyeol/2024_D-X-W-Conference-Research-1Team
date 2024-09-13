import torch
import torch.nn as nn
import torch.nn.functional as F

class UNET1D(nn.Module):
    def __init__(self, in_channels=40, out_channels=40):
        super(UNET1D, self).__init__()

        def CBR1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
            layers = [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            ]
            return nn.Sequential(*layers)

        # Contracting path
        self.cont1 = CBR1d(in_channels, 64, padding=1)
        self.cont2 = CBR1d(64, 64, padding=1)
        self.pool64 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.cont3 = CBR1d(64, 128, padding=1)
        self.cont4 = CBR1d(128, 128, padding=1)
        self.pool128 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.cont5 = CBR1d(128, 256, padding=1)
        self.cont6 = CBR1d(256, 256, padding=1)
        self.pool256 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.cont7 = CBR1d(256, 512, padding=1)
        self.cont8 = CBR1d(512, 512, padding=1)
        self.pool512 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.cont9 = CBR1d(512, 1024, padding=1)

        # Expanding path
        self.up512 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.exp1 = CBR1d(1024, 512, padding=1)
        self.exp2 = CBR1d(512, 512, padding=1)

        self.up256 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.exp3 = CBR1d(512, 256, padding=1)
        self.exp4 = CBR1d(256, 256, padding=1)

        self.up128 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.exp5 = CBR1d(256, 128, padding=1)
        self.exp6 = CBR1d(128, 128, padding=1)

        self.up64 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.exp7 = CBR1d(128, 64, padding=1)
        self.exp8 = CBR1d(64, 64, padding=1)

        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        con_x1 = self.cont1(x)
        con_x2 = self.cont2(con_x1)
        con_pool_1 = self.pool64(con_x2)

        con_x3 = self.cont3(con_pool_1)
        con_x4 = self.cont4(con_x3)
        con_pool_2 = self.pool128(con_x4)

        con_x5 = self.cont5(con_pool_2)
        con_x6 = self.cont6(con_x5)
        con_pool_3 = self.pool256(con_x6)

        con_x7 = self.cont7(con_pool_3)
        con_x8 = self.cont8(con_x7)
        con_pool_4 = self.pool512(con_x8)

        con_x9 = self.cont9(con_pool_4)

        # Expanding path
        exp_up_1 = self.up512(con_x9)
        exp_up_1 = self._match_dimensions(exp_up_1, con_x8)
        exp_up_1 = torch.cat([exp_up_1, con_x8], dim=1)
        exp_x1 = self.exp1(exp_up_1)
        exp_x2 = self.exp2(exp_x1)

        exp_up_2 = self.up256(exp_x2)
        exp_up_2 = self._match_dimensions(exp_up_2, con_x6)
        exp_up_2 = torch.cat([exp_up_2, con_x6], dim=1)
        exp_x3 = self.exp3(exp_up_2)
        exp_x4 = self.exp4(exp_x3)

        exp_up_3 = self.up128(exp_x4)
        exp_up_3 = self._match_dimensions(exp_up_3, con_x4)
        exp_up_3 = torch.cat([exp_up_3, con_x4], dim=1)
        exp_x5 = self.exp5(exp_up_3)
        exp_x6 = self.exp6(exp_x5)

        exp_up_4 = self.up64(exp_x6)
        exp_up_4 = self._match_dimensions(exp_up_4, con_x2)
        exp_up_4 = torch.cat([exp_up_4, con_x2], dim=1)
        exp_x7 = self.exp7(exp_up_4)
        exp_x8 = self.exp8(exp_x7)

        return self.final_conv(exp_x8)

    def _match_dimensions(self, x, target):
        if x.size(2) < target.size(2):
            diff = target.size(2) - x.size(2)
            x = F.pad(x, (diff // 2, diff - diff // 2))
        elif x.size(2) > target.size(2):
            diff = x.size(2) - target.size(2)
            x = x[:, :, diff // 2 : -(diff - diff // 2)]
        return x

if __name__ == "__main__":
    model = UNET1D()
    input_tensor = torch.randn(32, 40, 512)  # Changed to match your input size
    output = model(input_tensor)
    print(output.shape)