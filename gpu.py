import torch

print(torch.__version__)                    # Should show version+cu121 or similar
print(torch.cuda.is_available())            # Should print True
print(torch.cuda.device_count())            # Should print 1 (or more)
print(torch.cuda.get_device_name(0))        # Should print your GPU name

# Quick tensor test on GPU
x = torch.tensor([1.0, 2.0]).cuda()
print(x.device)  # Should print: cuda:0