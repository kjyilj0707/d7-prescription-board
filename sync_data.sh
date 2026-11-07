#!/bin/zsh
# 프로젝트 폴더에서 데이터 재생성(rebuild.sh) 후 이 저장소로 데이터·템플릿을 가져와 index.html 다시 만들기
P="/Users/yeobby/DartB_8th/DArt-B X Parrot_DataTone/outputs/mvp"
cd "$(dirname "$0")"
cp "$P/games_test.json" "$P/engine.json" data/ && python3 build.py && git status --short
