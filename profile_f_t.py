from thop import profile as pf_flop
import torchprof as pf_time
import pandas as DataFrame

def profile(model, inp_data, want_op_file=False, cuda_=False):
  df1 = pf_flop(model, inputs=(inp_data, ))  
  
  with pf_time.Profile(model, use_cuda=cuda_) as prof:
    model(inp_data)
  df2=prof.display()
  
  mynn={"Layer Name":[],"FLOPs":[],"Self CPU total":[], "CPU Total":[], "GPU Total":[],"Input Features":[], "Output Features":[], "Dict Size of Emb":[], "Emb Vector Size":[], "Norm Size":[]}
  for i1 in df1.index:
    mynn["Layer Name"][i1].append(row1["Layer Name"])
    mynn["Self CPU total"][i1].append(row1["Self CPU total"])
    mynn["CPU Total"][i1].append(row1["CPU total"])
    mynn["GPU Total"][i1].append(row1["GPU total"])
    mynn["Input Features"][i1].append(row2["Input Features"])
    mynn["Output Features"][i1].append(row2["Output Features"])
    mynn["Dict Size of Emb"][i1].append(row2["Dict Size of Emb"])
    mynn["Emb Vector Size"][i1].append(row2["Emb Vector Size"])
    mynn["Norm Size"][i1].append(row2["Norm Size"])
  
  df3 = DataFrame(mynn, columns= ["Layer Name","FLOPs","Self CPU total","CPU Total","GPU Total","Input Features","Output Features","Dict Size of Emb","Emb Vector Size","Norm Size"])
 
  if want_op_file==True:
    export_csv = df3.to_csv (r'output_file.csv', index = None, header=True)
  else:
    print(df3)
    
    
