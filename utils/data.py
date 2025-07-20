from __future__ import annotations

import json
import re
from collections.abc import Generator
from enum import Enum

import requests
from pydantic import BaseModel

from utils.constants import FacadPath


class FacadImage(BaseModel):
    color: str
    url: str
    image: bytes | None = None


class FacadData(BaseModel):
    id: int
    images: list[FacadImage]
    title: str
    description: str
    detail_info: str
    categoryid: int
    category: str
    attr: list[str]
    attrid: list[int]


class FacadDataList(BaseModel):
    data: list[FacadData]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> FacadData:
        return self.data[index]

    def __iter__(self) -> Generator[FacadData, None, None]:  # type: ignore
        yield from self.data

    @staticmethod
    def load_data(
        data_num: int, category_filter: list[ItemCategory] | ItemCategory | None = None
    ) -> FacadDataList:
        """
        画像ファイルを読み込み。画像が存在しない場合は、wgetでダウンロードして保存する。

        Args:
            image_num: 画像の取得数

        Returns:
            FacadDataList
        """
        master_json_path = FacadPath.MASTER_JSON_PATH
        with open(master_json_path) as f:
            data = json.load(f)

        filtered_data = []
        for item in data:
            if category_filter is None or item["category"] in (
                cat.value
                for cat in (
                    category_filter
                    if isinstance(category_filter, list)
                    else [category_filter]
                )
            ):
                filtered_data.append(item)
            if len(filtered_data) == data_num:
                break
        validated_data = FacadDataList(
            data=[
                FacadData(
                    id=item["id"],
                    images=[
                        FacadImage(
                            color=image["color"],
                            url=image["0"],
                            image=None,
                        )
                        for image in item["images"]
                    ],
                    title=item["title"],
                    description=item["description"],
                    detail_info=item["detail_info"],
                    categoryid=item["categoryid"],
                    category=item["category"],
                    attr=item["attr"],
                    attrid=item["attrid"],
                )
                for item in filtered_data
            ]
        )

        for item in validated_data:
            for idx, image in enumerate(item.images):
                image_path = FacadPath.IMAGE_PATH / f"{item.id}_{idx}.jpeg"
                if image_path.exists():
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    image.image = image_bytes
                else:
                    full_url = image.url
                    url_base = re.split(r"(?<=\.jpeg)", full_url)[0]
                    crop_str = "?crop=pad&pad_color=FFF&format=jpeg&w=512&h=512"
                    url_crop = url_base + crop_str
                    image_bytes = requests.get(url_crop).content
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    image.image = image_bytes

        return validated_data


class ItemCategory(Enum):
    """
    アイテムのカテゴリを定義するEnum
    """

    BACKPACK = "backpack"
    BAG = "bag"
    BELT = "belt"
    BLAZER = "blazer"
    BLOUSE = "blouse"
    BODYSUIT = "bodysuit"
    BOOT = "boot"
    BOTTOM = "bottom"
    BRA = "bra"
    BRACELET = "bracelet"
    BRALETTE = "bralette"
    BRIEF = "brief"
    CAMISOLE = "camisole"
    CARDIGAN = "cardigan"
    CASE = "case"
    CHEMISE = "chemise"
    CLUTCH = "clutch"
    COAT = "coat"
    DERBY = "derby"
    DRESS = "dress"
    EARRING = "earring"
    FLAT = "flat"
    FLOP = "flop"
    GLASS = "glass"
    GOWN = "gown"
    HAT = "hat"
    HENLEY = "henley"
    HOOD = "hood"
    JACKET = "jacket"
    JEANS = "jeans"
    JUMPSUIT = "jumpsuit"
    LEGGING = "legging"
    LOAFER = "loafer"
    MINIDRESS = "minidress"
    MINISKIRT = "miniskirt"
    MULE = "mule"
    NECKLACE = "necklace"
    ON = "on"
    OXFORD = "oxford"
    PAJAMAS = "pajamas"
    PANTS = "pants"
    PARKA = "parka"
    POLO = "polo"
    PULLOVER = "pullover"
    PUMP = "pump"
    RING = "ring"
    ROBE = "robe"
    ROMPER = "romper"
    SANDAL = "sandal"
    SATCHEL = "satchel"
    SCARF = "scarf"
    SHIRTDRESS = "shirtdress"
    SHOE = "shoe"
    SHORTS = "shorts"
    SKIRT = "skirt"
    SLIPDRESS = "slipdress"
    SLIPPERS = "slippers"
    SNEAKER = "sneaker"
    SOCK = "sock"
    SUIT = "suit"
    SUNDRESS = "sundress"
    SUNGLASS = "sunglass"
    SWEATER = "sweater"
    SWEATPANTS = "sweatpants"
    SWEATSHIRT = "sweatshirt"
    SWIMSUIT = "swimsuit"
    TANK = "tank"
    TEE = "tee"
    THONGS = "thongs"
    TIE = "tie"
    TIGHTS = "tights"
    TOP = "top"
    TOTE = "tote"
    TROUSERS = "trousers"
    TRUNK = "trunk"
    TUNIC = "tunic"
    VEST = "vest"
    WALLET = "wallet"

    @staticmethod
    def get_all_categories() -> list[str]:
        """
        全てのカテゴリ名を取得

        Returns:
            カテゴリ名のリスト
        """
        return [category.name for category in ItemCategory]
