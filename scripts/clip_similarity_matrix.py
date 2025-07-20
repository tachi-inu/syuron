import io
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import axes, figure
from PIL import Image

from utils.constants import FacadPath
from utils.data import FacadDataList
from utils.models.clip import CLIPMixin

warnings.filterwarnings(action="ignore")

JST = timezone(timedelta(hours=+9), "JST")


class CLIPSimilarityMatrix(CLIPMixin):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        """
        CLIP類似度マトリックス作成クラス

        Args:
            model_name: 使用するCLIPモデル名
        """
        super().__init__(model_name)

    def compute_similarity_matrix(self, data: FacadDataList) -> np.ndarray:
        """
        テキストと画像の類似度マトリックスを計算

        Args:
            data_num: データの数

        Returns:
            類似度マトリックス (len(texts) x len(image_paths))
        """
        print("テキストをエンコード中...")
        text_features = self.encode_texts([item.title for item in data.data])
        print("画像をエンコード中...")
        image_features = self.encode_images([item.images[0] for item in data.data])

        print("類似度マトリックスを計算中...")
        # コサイン類似度を計算
        similarity_matrix = torch.matmul(text_features, image_features.T)

        return similarity_matrix.cpu().numpy()

    def visualize_similarity_matrix(
        self,
        similarity_matrix: np.ndarray,
        text_labels: list[str],
        images: list[bytes],
        figsize: tuple = (15, 10),
        image_height: float = 0.8,
        save_path: Path | None = None,
    ) -> None:
        """
        類似度マトリックスを可視化（横軸に画像を配置）

        Args:
            similarity_matrix: 類似度マトリックス
            text_labels: テキストラベルのリスト
            images: 画像のリスト
            figsize: 図のサイズ
            image_height: 画像の高さ（図の下部からの相対位置）
            save_path: 保存先パス（Noneの場合は表示のみ）
        """
        fig, ax = plt.subplots(figsize=figsize)

        # ヒートマップを描画
        im = ax.imshow(similarity_matrix, cmap="viridis", aspect="auto")

        # テキストラベルを設定(縦軸)
        ax.set_yticks(range(len(text_labels)))
        ax.set_yticklabels(text_labels, fontsize=10)

        # 横軸の設定
        ax.set_xticks(range(len(images)))
        ax.set_xticklabels(
            [f"Image {i + 1}" for i in range(len(images))], fontsize=8, rotation=45
        )

        # カラーバーを追加
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Similarity", rotation=270, labelpad=20, fontsize=12)

        # マトリックス内に数値を表示
        for i in range(len(text_labels)):
            for j in range(len(images)):
                ax.text(
                    j,
                    i,
                    f"{similarity_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    weight="bold",
                )

        # タイトルを設定
        ax.set_title(
            "CLIP Similarity Matrix (Text vs. Image)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        # 画像を横軸の下に配置
        self._add_images_to_axis(fig, ax, images, image_height)

        timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        if not save_path:
            save_path = FacadPath.OUTPUT_PATH / "similarity_matrix.png"

        save_path = save_path.with_name(
            f"{save_path.stem}_{timestamp}{save_path.suffix}"
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"結果を保存しました: {save_path}")
        plt.close(fig)

    def _add_images_to_axis(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        images: list[bytes],
        image_height: float = 0.8,
    ) -> None:
        """
        横軸の下に画像を配置する内部メソッド
        """
        # 軸の位置情報を取得
        pos = ax.get_position()
        image_width = pos.width / len(images)

        for i, image in enumerate(images):
            # 画像を読み込み
            img = Image.open(io.BytesIO(image)).convert("RGB")

            # サムネイルサイズに調整
            img.thumbnail((100, 100), Image.Resampling.LANCZOS)

            # 画像を配置する位置を計算
            left = pos.x0 + i * image_width + image_width * 0.1
            bottom = pos.y0 - 0.15
            width = image_width * 0.8
            height = 0.1

            # 軸を作成して画像を表示
            img_ax = fig.add_axes(rect=(left, bottom, width, height))
            img_ax.imshow(img)
            img_ax.axis("off")


# 使用例
def main() -> None:
    """
    使用例のメイン関数
    """

    print("=== CLIP類似度マトリックス作成 ===")
    data = FacadDataList.load_data(data_num=10)

    images = [item.images[0].image for item in data if item.images[0].image is not None]
    text_labels = [item.title for item in data.data]

    # CLIPSimilarityMatrixインスタンスを作成
    clip_matrix = CLIPSimilarityMatrix()

    # 類似度マトリックスを計算
    similarity_matrix = clip_matrix.compute_similarity_matrix(data)

    # 結果を可視化
    clip_matrix.visualize_similarity_matrix(
        similarity_matrix,
        text_labels,
        images,
        figsize=(15, 10),
        save_path=FacadPath.OUTPUT_PATH / "similarity_matrix.png",
    )


if __name__ == "__main__":
    main()
