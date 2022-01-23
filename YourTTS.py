import os
# import requests
import gdown
import torch
import IPython
from IPython.display import Audio
from playsound import playsound
from TTS.config import load_config
from TTS.tts.models import setup_model
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis


class YourTTS(object):
    def __init__(self, ref_dir="./ref/", cfg_dir="./cfg/", out_dir="./out/"):
        self.ref_dir = ref_dir
        self.cfg_dir = cfg_dir
        self.out_dir = out_dir

        os.makedirs(ref_dir, exist_ok=True)
        os.makedirs(cfg_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        self.model_path = os.path.join(self.cfg_dir, "best_model.pth.tar")
        self.config_path = os.path.join(self.cfg_dir, "config.json")
        self.language_path = os.path.join(self.cfg_dir, "language_ids.json")
        self.speakers_path = os.path.join(self.cfg_dir, "speakers.json")
        self.checkpoint_path = os.path.join(self.cfg_dir, "SE_checkpoint.pth.tar")
        self.encoder_config_path = os.path.join(self.cfg_dir, "encoder_config.json")

        self.paths = {
            self.model_path:
                "https://drive.google.com/u/0/uc?export=download&confirm=wj9s&id=1KzdNYdtzsdXq9rbZZmKNbUEK2rBarklc",
            self.config_path:
                "https://drive.google.com/u/0/uc?id=1eeQN46gTLyMG7xiDvf3m7R7Nf1iO0lqZ&export=download",
            self.language_path:
                "https://drive.google.com/u/0/uc?id=1hB6_mXjoVIkllStFysWdJ9HFglzahluh&export=download",
            self.speakers_path:
                "https://drive.google.com/u/0/uc?id=1UIfU-a0V1NFz4V1IQMBryztZUn7s_-3d&export=download",
            self.checkpoint_path:
                "https://drive.google.com/u/0/uc?id=1fo1E8X39h4YAmbbXNGK3FsSjfSePKTzX&export=download",
            self.encoder_config_path:
                "https://drive.google.com/u/0/uc?id=1-4IdAZg1Xa_sc1O7VNWxsJd1FOe3Fx_w&export=download",
            os.path.join(self.ref_dir, "Kinda.wav"):
                "https://drive.google.com/u/0/uc?id=1n9-9jlhjL2ITXmw2YpcEVCWpN8A2vcze&export=download",
        }
        for k, v in self.paths.items():
            if not os.path.exists(k):
                # print("download file:", k)
                # req = requests.get(v, allow_redirects=True)
                # with open(k, 'wb') as f:
                #     f.write(req.content)
                gdown.download(v, k)

        self.use_cuda = torch.cuda.is_available()
        self.config = load_config(self.config_path)
        self.config["model_args"]["d_vector_file"] = self.speakers_path
        self.config["model_args"]["use_speaker_encoder_as_loss"] = False

        self.model = setup_model(self.config)
        self.model.language_manager.set_language_ids_from_file(self.language_path)
        cp = torch.load(self.model_path, map_location=torch.device('cpu'))
        model_weights = cp["model"].copy()
        for key in list(model_weights.keys()):
            if "speaker_encoder" in key:
                del model_weights[key]
        self.model.load_state_dict(model_weights)
        self.model.eval()
        if self.use_cuda: self.model = self.model.cuda()

        self.ap = AudioProcessor(**self.config["audio"])
        SE_speaker_manager = SpeakerManager(
            encoder_model_path=self.checkpoint_path,
            encoder_config_path=self.encoder_config_path,
            use_cuda=self.use_cuda
        )
        reference_files = [os.path.join(self.ref_dir, x) for x in os.listdir(self.ref_dir)]
        self.reference_emb = SE_speaker_manager.compute_d_vector_from_clip(reference_files)

    def say(self, text, language_id=0,
            length_scale=1,
            inference_noice_scale=0.3,
            inference_noise_scale_dp=0.3,
            filename="a.wav"):
        # scaler for the duration predictor. The larger it is, the slower the speech.
        self.model.length_scale = length_scale
        # defines the noise variance applied to the random z vector at inference.
        self.model.inference_noise_scale = inference_noice_scale
        # defines the noise variance applied to the duration predictor z vector at inference.
        self.model.inference_noise_scale_dp = inference_noise_scale_dp

        wav, alignment, _, _ = synthesis(
            self.model, text,
            self.config, self.use_cuda, self.ap,
            d_vector=self.reference_emb, language_id=language_id,
            enable_eos_bos_chars=self.config["enable_eos_bos_chars"],
            use_griffin_lim=True, do_trim_silence=False,
        ).values()
        IPython.display.display(Audio(wav, rate=self.ap.sample_rate))
        out_path = os.path.join(self.out_dir, filename)
        self.ap.save_wav(wav, out_path)
        playsound(out_path)


if __name__ == "__main__":
    _speaker = YourTTS(cfg_dir="./cfg2/")
    _speaker.say("I am ready")

    while True:
        _text = input("INPUT:")
        _speaker.say(_text)
        if "bye" in _text: break
    print("END")
