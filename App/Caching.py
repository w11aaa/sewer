from Library import App, Torch, Hashlib, OS, YOLO

@App.cache_resource(show_spinner=False)
def BuildModel(Content, Extension, Dir, device_preference='auto'):
    if Extension.upper() == ".ENGINE" and not Torch.cuda.is_available():
        App.toast("TensorRT只能在NVIDIA硬件上运行", icon="❌")
        return None
    
    # 检查设备偏好设置
    if Extension.upper() == ".ENGINE" and device_preference == 'cpu':
        App.toast("TensorRT模型无法在CPU上运行，将自动使用GPU", icon="⚠️")
        device_preference = 'cuda'
    
    ModelHash = Hashlib.md5(Content).hexdigest()
    ModelName = f"YOLOMODEL@{ModelHash.upper()}"
    HashPath = OS.path.join(Dir, f"{ModelName}{Extension}")
    with open(HashPath, "wb") as Fl:
        Fl.write(Content)
    
    # 创建模型
    NewM = YOLO(HashPath, "segment")
    
    # 根据设备偏好设置模型设备
    if device_preference == 'cpu':
        NewM.to('cpu')
    elif device_preference == 'cuda' and Torch.cuda.is_available():
        NewM.to('cuda')
    elif device_preference == 'auto':
        # 自动选择：优先GPU，不可用则CPU
        if Torch.cuda.is_available():
            NewM.to('cuda')
        else:
            NewM.to('cpu')
    
    # 预热模型
    GetSample = (1, 3, 640, 640)
    device = next(NewM.model.parameters()).device
    NewM(Torch.zeros(GetSample).to(device))
    
    return NewM