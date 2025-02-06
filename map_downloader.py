from PIL import Image
import tifffile

# TIF 파일 열기 및 PNG로 저장
tif_file_path = "Sentinel2_S2.tif"
png_file_path = "Sentinel2_S2.png"

# tifffile로 파일 확인
try:
    with tifffile.TiffFile(tif_file_path) as tif:
        print(f"TIFF 정보: {tif.pages}")
except Exception as e:
    print(f"TIFF 파일 읽기 실패: {e}")

# Pillow로 TIFF 파일 열기
try:
    with Image.open(tif_file_path) as img:
        img.save(png_file_path, format="PNG")
        print(f"변환 완료: {png_file_path}")
except Exception as e:
    print(f"이미지 변환 실패: {e}")
