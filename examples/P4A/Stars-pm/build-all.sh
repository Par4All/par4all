#!/bin/bash


for SIZE in 32 64 128 ; do
  make build-all SIZE=${SIZE} opengl=1
  make build-all SIZE=${SIZE} gtk=1
  make build-all SIZE=${SIZE} timing=1
done

