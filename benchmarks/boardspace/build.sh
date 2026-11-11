# bash benchmarks/boardspace/build.sh   -> $SHM/classes (run with $WORK/jdk/bin/java -cp $SHM/classes gipf.BsBridge <strategy> <seed>)
# Fetches a portable JDK and the needed BoardSpace sources (GPL-3, not kept in this repo) into scratch and compiles
# GipfPlay + everything it reaches, the stubs in stubs/ and BsBridge.java. Nothing is installed.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${BOARDSPACE_WORK:-/workspace/tmp/gipf-work/external/boardspace}   # the JDK must sit on a filesystem that allows exec
SHM=${BOARDSPACE_SHM:-/dev/shm/boardspace}
COMMIT=0cdd3cbd026e1897784952e2daf39a5de5c11d59
mkdir -p "$WORK/jdk" "$SHM/src" "$SHM/classes"
if [ ! -x "$WORK/jdk/bin/javac" ]; then
  curl -sL -o "$SHM/jdk.tar.gz" https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse
  tar -xzf "$SHM/jdk.tar.gz" -C "$WORK/jdk" --strip-components=1 --exclude='*/jmods' --exclude='*/lib/src.zip' --exclude='*/man' --exclude='*/legal'
fi
if [ ! -d "$SHM/src/repo" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/ddyer0/boardspace.net "$SHM/src/repo"
  git -C "$SHM/src/repo" sparse-checkout set --no-cone '/client/boardspace-java/boardspace-core/**/*.java' '/client/boardspace-java/boardspace-games/gipf/*.java' '/client/eclipse/jzlib/src/jzlib/*.java'
fi
[ "$(git -C "$SHM/src/repo" rev-parse HEAD)" = $COMMIT ] || echo "warning: BoardSpace is no longer at $COMMIT, the stubs may not fit" >&2
C="$SHM/src/repo/client"
"$WORK/jdk/bin/javac" -nowarn -encoding ISO-8859-1 -proc:none -d "$SHM/classes" \
  -sourcepath "$HERE/stubs:$C/boardspace-java/boardspace-core:$C/boardspace-java/boardspace-games:$C/eclipse/jzlib/src" \
  "$C/boardspace-java/boardspace-games/gipf/GipfPlay.java" "$C/boardspace-java/boardspace-core/bridge/InstallerPackageImpl.java" "$HERE/BsBridge.java"
