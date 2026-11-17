#!/bin/sh
set -eu
launcher=$(/usr/bin/readlink -f "$0")
exec "${launcher%/*}/nanbeige" chat "$@"
