@echo off
rem Збірка бенчмарку кешу (Windows / MinGW-w64 GCC)
rem Потрібен gcc у PATH або змінна GCC із повним шляхом.
if "%GCC%"=="" set GCC=gcc
%GCC% -O3 -march=native -Wall -Wextra -std=gnu11 -o cache_bench.exe cache_bench.c
if errorlevel 1 (
    echo Помилка збірки!
    exit /b 1
)
echo OK: cache_bench.exe
