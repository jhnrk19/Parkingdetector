#!/usr/bin/env bash
# Előkészíti a parkolóhely-detektáló szerver konténerét: ellenőriz, létrehozza a mappát és a .env fájlt.
set -euo pipefail
cd "$(dirname "$0")"

fail() { echo "HIBA: $*" >&2; exit 1; }

[ "$(pwd)" = "$HOME/parkolo" ] || fail "a projektnek itt kell lennie: $HOME/parkolo (most: $(pwd))"
docker info >/dev/null 2>&1 || fail "a docker nem érhető el ezzel a felhasználóval (benne vagy a docker csoportban?)"
docker compose version >/dev/null 2>&1 || fail "hiányzik a docker compose (sudo apt install docker-compose-plugin)"

mkdir -p data

image="nvcr.io/nvidia/pytorch:25.09-py3"
port="8000"
if [ -f .env ]; then
  oldimg="$(grep -E '^PK_PYTORCH_IMAGE=' .env | cut -d= -f2- || true)"; [ -z "$oldimg" ] || image="$oldimg"
  oldport="$(grep -E '^PK_PORT=' .env | cut -d= -f2- || true)"; [ -z "$oldport" ] || port="$oldport"
fi

tz="$(cat /etc/timezone 2>/dev/null || timedatectl show -p Timezone --value 2>/dev/null || true)"
[ -n "$tz" ] || tz="Europe/Budapest"

# Van-e NVIDIA videókártya a dockerben? (A kimenetet előbb változóba tesszük: "pipefail" mellett a
# "docker info | grep -q" akkor is hamisat adna, ha van találat.)
has_gpu=0
dinfo="$(docker info 2>/dev/null || true)"
if grep -qi 'nvidia' <<<"$dinfo" || command -v nvidia-smi >/dev/null 2>&1 || [ -e /etc/cdi/nvidia.yaml ] || [ -e /var/run/cdi/nvidia.yaml ]; then has_gpu=1; fi
[ "${PK_GPU:-}" = "1" ] && has_gpu=1
# PK_GPU=0 ./docker-setup.sh  -> videókártya nélkül (kis konténer, lassú)
compose="docker-compose.yml"
gpu="nem (processzoron fut, lassú)"
if [ "${PK_GPU:-1}" != "0" ] && [ "$has_gpu" = "1" ]; then
  compose="docker-compose.yml:docker-compose.gpu.yml"
  gpu="igen"
fi

cat > .env <<ENV
# A docker-setup.sh hozta létre. Újrafuttatáskor a port és az alapkép megmarad.
COMPOSE_FILE=$compose
PK_UID=$(id -u)
PK_GID=$(id -g)
HOST_HOME=$HOME
# A kliens a http://<gép címe>:8000 címet várja; csak akkor írd át, ha a kliensben is átállítod.
PK_PORT=$port
PK_TZ=$tz
# Ennyi óráig maradnak meg az eredményvideók (a megtanult munkamenetek nem törlődnek):
PK_KEEP_HOURS=72
# A videókártyás változat alapképe (az NVIDIA PyTorch-konténere):
PK_PYTORCH_IMAGE=$image
ENV
chmod 600 .env

if grep -q ":$port " <<<"$(ss -ltn 2>/dev/null || true)"; then
  echo "FIGYELEM: a(z) $port-es portot már használja valami ezen a gépen (ss -ltnp | grep :$port)."
  echo "          Állítsd le azt, vagy írd át a PK_PORT-ot a .env-ben és a kliensben is."
fi

echo "Kész. Videókártya: $gpu"
[ "$gpu" = "igen" ] && echo "Az első építés letöltheti az NVIDIA PyTorch-konténerét (kb. 15-20 GB), ha még nincs a gépen."
echo "Indítás: docker compose up -d --build"
