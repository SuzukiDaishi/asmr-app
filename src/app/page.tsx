"use client";

import { invoke } from '@tauri-apps/api/tauri'
import ASMRGenerator from '@/components/AsmrGenerator'

export default function Home() {
  return (
    <ASMRGenerator invoke={invoke} />
  )
}
