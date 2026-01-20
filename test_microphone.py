#!/usr/bin/env python3
"""
Microphone diagnostic tool for WebRTC audio client
Tests different audio sources to find what works on your system
"""

import sys
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from aiortc.contrib.media import MediaPlayer
except ImportError:
    print("ERROR: aiortc not installed. Run: pip install aiortc")
    sys.exit(1)


async def test_audio_source(source, format_name, options=None):
    """Test if an audio source works"""
    options = options or {}
    try:
        logger.info(f"Testing: source='{source}', format='{format_name}', options={options}")
        player = MediaPlayer(source, format=format_name, options=options)

        # Try to read a few frames
        for i in range(5):
            frame = await player.audio.recv()
            if i == 0:
                logger.info(f"  ✅ SUCCESS! Frame info: samples={frame.samples}, "
                          f"rate={frame.sample_rate}, format={frame.format.name}")

        return True
    except Exception as e:
        logger.warning(f"  ❌ Failed: {e}")
        return False


async def main():
    print("=" * 70)
    print("WebRTC Microphone Diagnostic Tool")
    print("=" * 70)
    print(f"Platform: {sys.platform}")
    print()

    if sys.platform == "darwin":
        print("macOS Audio Sources:")
        print("-" * 70)
        sources = [
            (":0", "avfoundation", {"sample_rate": "48000"}),
            (":default", "avfoundation", {"sample_rate": "48000"}),
            ("0", "avfoundation", {"sample_rate": "48000"}),
            (":0", "avfoundation", {}),
        ]

    elif sys.platform.startswith("linux"):
        print("Linux Audio Sources:")
        print("-" * 70)

        # First, give some diagnostic info
        import subprocess

        print("\n1. Checking PulseAudio:")
        try:
            result = subprocess.run(["pactl", "info"], capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✅ PulseAudio is running")
            else:
                print("   ❌ PulseAudio not available")
        except FileNotFoundError:
            print("   ❌ pactl command not found (PulseAudio not installed)")

        print("\n2. Checking ALSA devices:")
        try:
            result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("   ❌ No ALSA devices found")
        except FileNotFoundError:
            print("   ❌ arecord command not found (ALSA not installed)")

        print("\n3. Testing audio sources:")
        print("-" * 70)

        sources = [
            ("default", "pulse", {"sample_rate": "48000"}),
            ("0", "pulse", {}),
            ("default", "alsa", {"sample_rate": "48000", "channels": "1"}),
            ("hw:0,0", "alsa", {"sample_rate": "48000", "channels": "1"}),
            ("hw:0", "alsa", {"sample_rate": "48000", "channels": "1"}),
            ("plughw:0,0", "alsa", {"sample_rate": "48000", "channels": "1"}),
        ]

    else:
        print("Windows Audio Sources:")
        print("-" * 70)
        sources = [
            ("audio=Microphone", "dshow", {}),
        ]

    # Test each source
    working_sources = []
    for source, fmt, options in sources:
        result = await test_audio_source(source, fmt, options)
        if result:
            working_sources.append((source, fmt, options))
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if working_sources:
        print(f"✅ Found {len(working_sources)} working audio source(s):")
        for source, fmt, options in working_sources:
            print(f"   - source='{source}', format='{fmt}', options={options}")
        print("\nYour microphone is working! If the WebRTC client still has issues,")
        print("the problem is likely with permissions or audio playback.")
    else:
        print("❌ No working audio sources found!")
        print("\nTroubleshooting steps:")

        if sys.platform == "darwin":
            print("1. Grant microphone permissions:")
            print("   System Preferences → Security & Privacy → Privacy → Microphone")
            print("2. Test with: ffmpeg -f avfoundation -list_devices true -i \"\"")

        elif sys.platform.startswith("linux"):
            print("1. Install required packages:")
            print("   sudo apt-get install alsa-utils pulseaudio ffmpeg")
            print("2. Test microphone:")
            print("   arecord -d 3 test.wav && aplay test.wav")
            print("3. Check PulseAudio:")
            print("   pactl list sources short")
            print("4. Check permissions:")
            print("   ls -l /dev/snd/")


if __name__ == "__main__":
    asyncio.run(main())
