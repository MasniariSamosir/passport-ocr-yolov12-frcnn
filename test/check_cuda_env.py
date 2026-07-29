import torch
import platform
import subprocess
import os

def check_cuda_env():
    print("🚀 Memeriksa lingkungan CUDA dan PyTorch...\n")

    # 1️⃣ Cek OS dan Python
    print(f"🧩 Sistem Operasi : {platform.system()} {platform.release()}")
    print(f"🐍 Python Versi   : {platform.python_version()}")

    # 2️⃣ Cek versi PyTorch
    try:
        print(f"🔥 Torch Versi    : {torch.__version__}")
    except Exception as e:
        print(f"❌ PyTorch belum terinstal: {e}")
        return

    # 3️⃣ Cek apakah CUDA tersedia
    cuda_available = torch.cuda.is_available()
    print(f"⚙️ CUDA Tersedia  : {cuda_available}")

    if cuda_available:
        print(f"💻 GPU Terdeteksi : {torch.cuda.get_device_name(0)}")
        print(f"🧠 CUDA Versi Torch: {torch.version.cuda}")
        print(f"📦 Jumlah GPU     : {torch.cuda.device_count()}")
        print(f"📊 Memori GPU (MB): {round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2)} MB")
    else:
        print("\n⚠️  CUDA tidak aktif di PyTorch.")
        print("   Penyebab umum:")
        print("   • CUDA Toolkit belum terinstal atau tidak sesuai dengan Torch.")
        print("   • Torch versi CPU-only (tanpa dukungan CUDA).")
        print("   • Driver NVIDIA belum terinstal atau tidak aktif.\n")

        # Coba cek driver NVIDIA dengan nvidia-smi
        try:
            result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                print("✅ Driver NVIDIA aktif (nvidia-smi tersedia):")
                print(result.stdout)
            else:
                print("❌ Tidak dapat menjalankan nvidia-smi (Driver NVIDIA belum aktif atau tidak ada GPU).")
        except FileNotFoundError:
            print("❌ Perintah nvidia-smi tidak ditemukan. Pastikan driver NVIDIA dan CUDA Toolkit sudah diinstal.\n")
    
    print("\n🔍 Selesai memeriksa lingkungan sistem.")
    print("Jika CUDA belum aktif, jalankan langkah berikut:")
    print("👉 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n")
    print("Pastikan juga CUDA Toolkit versi 12.1 sudah terpasang dari https://developer.nvidia.com/cuda-downloads\n")

if __name__ == "__main__":
    check_cuda_env()
