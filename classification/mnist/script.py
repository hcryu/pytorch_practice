import argparse

# ArgumentParser 객체 생성
parser = argparse.ArgumentParser()

# 명령줄 인자 추가
parser.add_argument("--device", default="cpu", help="device for training")

# 명령줄 인자 파싱
args = parser.parse_args()

# 결과 출력
print(f"Selected device: {args.device}")
print(args)