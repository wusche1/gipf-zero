# bash build.sh [harness dir]   -> <harness dir>/gf1bridge
# Needs the headless GF1 1.03 harness (GF1 sources in ../gf1/src, objects built by its own build.sh); GF1 itself is GPL and not kept in this repo.
H=${1:-/workspace/tmp/gipf-work/external/gf1-harness}
cd "$H" && g++ -O2 -w -fpermissive -std=gnu++98 -Ishim -I../gf1/src -include shim/prefix.h "$OLDPWD/gf1bridge.cxx" ai_gipf.o ai_player.o board2.o gamelog2.o xmlite.o configfile.o linklist.o position.o rem_gipf.o rem_row.o -o gf1bridge
