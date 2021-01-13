import torch
import torch.nn.functional as F

if __name__ == '__main__':
    t1 = torch.tensor([[1, 2]]).float()
    t3 = torch.tensor([1])
    t2 = F.cross_entropy(input=t1, target=t3, reduction="mean")
    print("t2 is", t2)
    t1_softmaxed = F.softmax(t1, dim=1)
    minus_log_t1_softmaxed = -torch.log(t1_softmaxed)
    # print("minus_log_t1_softmaxed is", minus_log_t1_softmaxed)
    print("minus_log_t1_softmaxed[1] is", minus_log_t1_softmaxed[0][1])
