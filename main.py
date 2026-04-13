from __future__ import annotations

import argparse
from typing import Dict, Optional

import librosa
import torch
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from model import TAESSERModel


class HuggingFaceAcousticEncoder(nn.Module):

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(model_name_or_path)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return {"last_hidden_state": outputs.last_hidden_state}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TAES-SER public review entry point")

    parser.add_argument("--mode", type=str, default="inspect", choices=["inspect", "single_audio"])
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, default=None)

    parser.add_argument("--num_emotions", type=int, default=4)
    parser.add_argument("--num_speakers", type=int, default=10)

    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--expert_bottleneck", type=int, default=256)
    parser.add_argument("--expert_dropout", type=float, default=0.1)

    parser.add_argument("--router_hidden", type=int, default=256)
    parser.add_argument("--router_dropout", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--router_temperature", type=float, default=2.0)
    parser.add_argument("--noisy_routing", action="store_true")

    parser.add_argument("--alpha_asr", type=float, default=0.1)
    parser.add_argument("--beta_sr", type=float, default=0.5)
    parser.add_argument("--mi_coef", type=float, default=0.0)
    parser.add_argument("--ent_coef", type=float, default=0.05)
    parser.add_argument("--balance_coef", type=float, default=0.03)

    return parser


def build_model_and_extractor(args: argparse.Namespace):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
    acoustic_encoder = HuggingFaceAcousticEncoder(args.model_name_or_path)

    hidden_size = acoustic_encoder.backbone.config.hidden_size
    vocab_size = acoustic_encoder.backbone.config.vocab_size

    model = TAESSERModel(
        acoustic_encoder=acoustic_encoder,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_emotions=args.num_emotions,
        num_speakers=args.num_speakers,
        num_experts=args.num_experts,
        expert_bottleneck=args.expert_bottleneck,
        expert_dropout=args.expert_dropout,
        router_hidden=args.router_hidden,
        router_dropout=args.router_dropout,
        top_k=args.top_k,
        router_temperature=args.router_temperature,
        noisy_routing=args.noisy_routing,
        alpha_asr=args.alpha_asr,
        beta_sr=args.beta_sr,
        mi_coef=args.mi_coef,
        ent_coef=args.ent_coef,
        balance_coef=args.balance_coef,
        use_masked_pooling=True,
    )
    return model, feature_extractor


def print_model_summary(args: argparse.Namespace) -> None:
    print("TAES-SER Public Review Configuration")
    print("=" * 60)
    print(f"backbone: {args.model_name_or_path}")
    print(f"num_emotions: {args.num_emotions}")
    print(f"num_speakers: {args.num_speakers}")
    print(f"num_experts: {args.num_experts}")
    print(f"top_k: {args.top_k}")
    print(f"router_temperature: {args.router_temperature}")
    print(f"alpha_asr: {args.alpha_asr}")
    print(f"beta_sr: {args.beta_sr}")
    print(f"mi_coef: {args.mi_coef}")
    print(f"ent_coef: {args.ent_coef}")
    print(f"balance_coef: {args.balance_coef}")
    print("=" * 60)
    print("Notes:")
    print("- SER is treated as the primary task.")
    print("- ASR and SR are auxiliary tasks.")
    print("- Full training, dataset protocol, and result reproduction remain private.")


def load_audio(audio_path: str, target_sr: int):
    speech, sr = librosa.load(audio_path, sr=target_sr)
    return speech, sr


def run_single_audio(args: argparse.Namespace) -> None:
    if not args.audio_path:
        raise ValueError("--audio_path is required when --mode single_audio is used.")

    model, feature_extractor = build_model_and_extractor(args)
    model.eval()

    speech, sr = load_audio(args.audio_path, feature_extractor.sampling_rate)
    inputs = feature_extractor(
        speech,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(
            input_values=inputs["input_values"],
            attention_mask=inputs.get("attention_mask", None),
            labels_ser=None,
            labels_asr=None,
            labels_sr=None,
            input_lengths=None,
            target_lengths=None,
            return_stats=True,
        )

    print("Single-audio forward completed.")
    print(f"audio_path: {args.audio_path}")
    print(f"logits_ser shape: {tuple(outputs.logits_ser.shape)}")
    print(f"logits_asr shape: {tuple(outputs.logits_asr.shape)}")
    print(f"logits_sr shape: {tuple(outputs.logits_sr.shape)}")

    if outputs.routing_stats is not None:
        print("\nRouting statistics")
        print("-" * 60)
        for key, value in outputs.routing_stats.items():
            print(f"{key}: {value:.6f}")

    if outputs.routing_choices is not None:
        print("\nTop-k expert indices")
        print("-" * 60)
        for task_name, task_info in outputs.routing_choices.items():
            print(f"{task_name}:")
            print(task_info["topk_idx"])


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "inspect":
        print_model_summary(args)
        return

    if args.mode == "single_audio":
        run_single_audio(args)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
