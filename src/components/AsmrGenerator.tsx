"use client";

import React, { useState, useEffect } from "react";
import { open } from "@tauri-apps/api/dialog";
import Knob from "../components/knob";
import ToggleSwitch from "./ToggleSwitch";
import { InvokeArgs } from "@tauri-apps/api/tauri";

type InvokeFunction = <T>(cmd: string, args?: InvokeArgs | undefined) => Promise<T>;

interface ASMRGeneratorProps {
  invoke: InvokeFunction;
}

const ASMRGenerator: React.FC<ASMRGeneratorProps> = ({ invoke }) => {
  const [filePath, setFilePath] = useState<string>("");
  const [preGainDb, setPreGainDb] = useState<number>(0);
  const [postGainDb, setPostGainDb] = useState<number>(0);
  const [azimuth, setAzimuth] = useState<number>(0);
  const [elevation, setElevation] = useState<number>(0);
  const [whisper, setWhisper] = useState<boolean>(true);
  const [getCloser, setGetCloser] = useState<boolean>(true);
  const [sourceNoise, setSourceNoise] = useState<string>("white");
  const [noiseOptions, setNoiseOptions] = useState<string[]>([]);

  // dB -> 線形倍率 変換 (dBが-60dB以下なら0にする)
  const dBToLinear = (db: number): number => (db <= -60 ? 0 : Math.pow(10, db / 20));

  useEffect(() => {
    invoke("get_noise_types")
      .then((types) => setNoiseOptions(types as string[]))
      .catch(console.error);
  }, [invoke]);

  // Azimuth/Elevation の Canvas 描画
  useEffect(() => {
    const canvas = document.getElementById("asmrCanvas") as HTMLCanvasElement | null;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#f0f0f0";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw head
        ctx.beginPath();
        ctx.arc(canvas.width / 2, canvas.height / 2, 50, 0, 2 * Math.PI);
        ctx.fillStyle = "#ccc";
        ctx.fill();

        // Draw sound source based on azimuth and elevation
        const radius = Math.min(canvas.width, canvas.height) / 3;
        const x = canvas.width / 2 + radius * Math.cos((azimuth * Math.PI) / 180);
        const y = canvas.height / 2 - radius * Math.sin((elevation * Math.PI) / 180);
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, 2 * Math.PI);
        ctx.fillStyle = "#f00";
        ctx.fill();
      }
    }
  }, [azimuth, elevation]);

  const handlePreGainChange = (newDb: number): void => {
    setPreGainDb(newDb);
    invoke("set_pre_gain", { gain: dBToLinear(newDb) }).catch(console.error);
  };

  const handlePostGainChange = (newDb: number): void => {
    setPostGainDb(newDb);
    invoke("set_post_gain", { gain: dBToLinear(newDb) }).catch(console.error);
  };

  const handleAzimuthChange = (newValue: number): void => {
    setAzimuth(newValue);
    invoke("set_hrtf_azimuth", { azimuth: newValue }).catch(console.error);
  };

  const handleElevationChange = (newValue: number): void => {
    setElevation(newValue);
    invoke("set_hrtf_elevation", { elevation: newValue }).catch(console.error);
  };

  const handlePlay = (): void => {
    console.log("Play audio with settings:", {
      filePath,
      preGainDb,
      postGainDb,
      azimuth,
      elevation,
      whisper,
      getCloser,
      sourceNoise,
    });
    // 引数のキー名を変更して、Tauri 側の期待に合わせる
    invoke("asmr_play", {
      path: filePath,
      whisper,
      sourceNoise, // ここを変更
      getCloser,   // もしこちらも要求されているなら変更
    }).catch(console.error);
  };
  

  const handleDryPlay = (): void => {
    invoke("dry_play", { path: filePath }).catch(console.error);
  };

  const handleStop = (): void => {
    invoke("stop_audio").catch(console.error);
  };

  const handleFileSelect = async (): Promise<void> => {
    const selectedFile = await open({
      filters: [{ name: "Audio Files", extensions: ["wav", "mp3", "flac"] }],
    });
    if (typeof selectedFile === "string") {
      setFilePath(selectedFile);
    }
  };

  return (
    <div className="flex flex-col items-center p-4 h-screen box-border select-none">
      <h1 className="text-2xl font-bold mb-4 select-none">ASMR Generator</h1>

      {/* ファイル選択 */}
      <div className="mb-4 flex flex-col items-start space-y-2 w-full max-w-6xl">
        <label className="text-sm font-semibold select-none">Audio File</label>
        <div className="flex items-center w-full">
          <button
            onClick={handleFileSelect}
            tabIndex={-1}
            className="px-4 py-2 border border-gray-300 nm-flat-gray-200 hover:nm-flat-gray-100 active:nm-inset-gray-200 rounded cursor-pointer mr-4"
          >
            Select File
          </button>
          <input
            type="text"
            className="w-full p-4 text-gray-600 bg-gray-200 nm-inset-gray-200 rounded-lg focus:outline-none select-text"
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            placeholder="No file selected"
          />
        </div>
      </div>

      <div className="p-4 border border-gray-300 rounded-lg w-full max-w-6xl h-[calc(100%-200px)] flex flex-col">
        <div className="flex justify-between flex-1 mb-4">
          {/* 左パネル */}
          <div className="flex flex-col w-1/4 space-y-4">
            <label>Pre Gain (dB)</label>
            <Knob
              size={80}
              mx={200}
              my={200}
              minValue={-60}
              maxValue={6}
              value={preGainDb}
              onChange={handlePreGainChange}
            />
            <label>Post Gain (dB)</label>
            <Knob
              size={80}
              mx={200}
              my={200}
              minValue={-60}
              maxValue={6}
              value={postGainDb}
              onChange={handlePostGainChange}
            />
            <label>Whisper</label>
            <ToggleSwitch checked={whisper} onChange={setWhisper} />
            <label>Source Noise</label>
            <select
              value={sourceNoise}
              onChange={(e) => setSourceNoise(e.target.value)}
              className="nm-flat-gray-200-xs hover:nm-flar-gray-100 active:nm-inset-gray-200 rounded-lg appearance-none w-full px-8 py-4 font-semibold select-text"
            >
              {noiseOptions.length > 0 ? (
                noiseOptions.map((opt) => (
                  <option key={opt} value={opt.toLowerCase()}>
                    {opt}
                  </option>
                ))
              ) : (
                <>
                  <option>Pink Noise</option>
                  <option>White Noise</option>
                  <option>Brown Noise</option>
                </>
              )}
            </select>
            <label>Get Closer</label>
            <ToggleSwitch checked={getCloser} onChange={setGetCloser} />
          </div>

          {/* 中央キャンバス */}
          <div className="flex-1 mx-4">
            <canvas
              id="asmrCanvas"
              className="w-full h-full border border-gray-300 rounded-lg"
              tabIndex={-1}
            ></canvas>
          </div>

          {/* 右パネル */}
          <div className="flex flex-col w-1/4 space-y-4">
            <label>Azimuth</label>
            <Knob
              size={80}
              mx={200}
              my={200}
              minValue={0}
              maxValue={360}
              value={azimuth}
              onChange={handleAzimuthChange}
            />
            <label>Elevation</label>
            <Knob
              size={80}
              mx={200}
              my={200}
              minValue={-90}
              maxValue={90}
              value={elevation}
              onChange={handleElevationChange}
            />
          </div>
        </div>

        {/* ボタン類 */}
        <div className="flex justify-center gap-4">
          <button onClick={handlePlay} tabIndex={-1} className="nm-button">
            Play
          </button>
          <button onClick={handleDryPlay} tabIndex={-1} className="nm-button">
            Dry Play
          </button>
          <button onClick={handleStop} tabIndex={-1} className="nm-button">
            Stop
          </button>
        </div>
      </div>
    </div>
  );
};

export default ASMRGenerator;
