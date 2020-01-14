from thop import profile as pf_flop
import torchprof as pf_time
import pandas as DataFrame

def profile(model, inp_data, want_op_file=False, cuda_=False):
  df1 = pf_flop(model, inputs=(inp_data, ))  
  data, targets = get_batch(val_data, 0)
  with pf_time.Profile(model, use_cuda=cuda_) as prof:
    model(inp_data)
  df2=prof.display()
  print(df1)
  print(df2)
