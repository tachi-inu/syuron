from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from utils.constants import FacadPath
from utils.data import FacadDataList, ItemCategory
from utils.models.clip import CLIPMixin


class DimensionReductionMethod:
    PCA = "pca"
    MDS = "mds"


class FashionMapVisualizer(CLIPMixin):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        """
        ファッションアイテムの画像をPCAで可視化するクラス

        Args:
            model_name: 使用するCLIPモデル名
        """
        super().__init__(model_name)
        self.data: FacadDataList | None = None
        self.features: np.ndarray | None = None
        self.reduced_result: np.ndarray | None = None
        self.reduction_method: str = DimensionReductionMethod.PCA

    def load_data(
        self, data_num: int, category: ItemCategory | list[ItemCategory] | None = None
    ) -> None:
        """
        指定したカテゴリのアイテムをロード

        Args:
            data_num: 取得するアイテム数
            category: アイテムのカテゴリ
        """
        self.data = FacadDataList.load_data(data_num, category)

    def extract_features(self) -> None:
        """
        CLIPで画像の特徴量を抽出
        """
        if self.data is None:
            raise ValueError(
                "データがロードされていません。load_data()を実行してください。"
            )

        all_images = []
        for item in self.data:
            all_images.extend(item.images)

        features = self.encode_images(all_images)
        self.features = features.cpu().numpy()

    def apply_dimension_reduction(
        self, method: str = DimensionReductionMethod.PCA, n_components: int = 2
    ) -> None:
        """
        次元削減を実行

        Args:
            method: 次元削減手法 ("pca" または "mds")
            n_components: 次元数
        """
        if self.features is None:
            raise ValueError(
                "特徴量が抽出されていません。extract_features()を実行してください。"
            )

        self.reduction_method = method
        if method == DimensionReductionMethod.PCA:
            reducer = PCA(n_components=n_components)
            self.reduced_result = reducer.fit_transform(self.features)
        elif method == DimensionReductionMethod.MDS:
            reducer = MDS(n_components=n_components, metric=True, n_jobs=-1)
            self.reduced_result = reducer.fit_transform(self.features)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

    def visualize(self, remark: str | None = None, thumbnail_size: float = 0.1) -> None:
        """
        次元削減の結果を可視化し、各点に対応する画像を表示

        Args:
            remark: 出力ファイル名の接頭辞
            thumbnail_size: サムネイル画像のサイズ（相対値）
        """
        if self.data is None:
            raise ValueError(
                "データがロードされていません。load_data()を実行してください。"
            )
        if self.reduced_result is None:
            raise ValueError(
                "次元削減が実行されていません。apply_dimension_reduction()を実行してください。"
            )

        fig, ax = plt.subplots(figsize=(15, 12))

        # 散布図の基本プロット
        ax.scatter(self.reduced_result[:, 0], self.reduced_result[:, 1], alpha=0)

        # 各点に画像を表示
        all_images = []
        for item in self.data:
            all_images.extend(item.images)

        for idx, (x, y) in enumerate(self.reduced_result):
            img_bytes = all_images[idx].image
            if img_bytes is not None:
                img = Image.open(io.BytesIO(img_bytes))
                # 画像をnumpy配列に変換
                img_array = np.array(img)
                # 画像を表示用に追加
                imagebox = OffsetImage(img_array, zoom=thumbnail_size)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                ax.add_artist(ab)

        method_name = (
            "PCA" if self.reduction_method == DimensionReductionMethod.PCA else "MDS"
        )
        plt.title(f"Fashion Items {method_name} Visualization with Images")
        plt.xlabel(f"First {method_name} Component")
        plt.ylabel(f"Second {method_name} Component")

        suffix = (
            remark + f"_{self.reduction_method}.png"
            if remark
            else f"{self.reduction_method}.png"
        )
        output_path = FacadPath.OUTPUT_PATH / suffix
        plt.savefig(output_path, dpi=300, bbox_inches="tight")


def main() -> None:
    # 使用例
    visualizer = FashionMapVisualizer()

    # PCAを使用した可視化
    visualizer.load_data(data_num=100, category=ItemCategory.DRESS)
    visualizer.extract_features()
    visualizer.apply_dimension_reduction(method=DimensionReductionMethod.PCA)
    visualizer.visualize(remark="dress_items_pca", thumbnail_size=0.05)

    # MDSを使用した可視化
    visualizer.apply_dimension_reduction(method=DimensionReductionMethod.MDS)
    visualizer.visualize(remark="dress_items_mds", thumbnail_size=0.05)


if __name__ == "__main__":
    main()
