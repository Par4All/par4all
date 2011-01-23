#!/bin/bash


for SIZE in 32 64 128 ; do
  make build-all SIZE=${SIZE} bench=1 # add here option like timing=1 do_not_cheat=1 gtk=1 opengl=1 oldgpu=1 debug=1
done

