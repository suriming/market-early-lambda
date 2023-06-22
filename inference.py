import torch
from model import LSTMEncoderDecoder

input_dim = 10
hidden_dim = 64
latent_dim = 8

class Inference:
    def __init__(self, model_path, device):
        self.device = device
        self.model_path = model_path
        #initialize model
        self.model = LSTMEncoderDecoder(input_dim, hidden_dim, latent_dim)
        
        #load model weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)

    def autoencoder_anomaly(self, input_data):
        # log = str(input_data)
        # log = " ".join(log.split())
        self.model.eval()
        with torch.no_grad():
            reconstructions, _ = self.model(input_data)
        # reconstruction_errors = 0.0

        reconstruction_errors = torch.mean((reconstructions - input_data) ** 2, dim=(1, 2))
        # print("reconstruction_errors:",reconstruction_errors)
        return reconstruction_errors

if __name__ == "__main__":
    MODEL_PATH = 'model/A030210.pth'
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    infer = Inference(MODEL_PATH, DEVICE)
    
    print("여기 오면 안ㅚ는데")