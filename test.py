import os, csv, subprocess

os.makedirs("Normal", exist_ok=True)
os.makedirs("Abnormal", exist_ok=True)

with open("ground_truth.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["no", "file_name", "property"])

    for label in ["Normal", "Abnormal"]:
        duration = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", f"{label}.MOV"
        ]).decode().strip())

        chunk = duration / 50

        for i in range(50):
            start = i * chunk
            output_name = f"{label}_{i+1:03d}.mov"
            output_path = os.path.join(label, output_name)
            subprocess.run([
                "ffmpeg", "-ss", str(start), "-i", f"{label}.MOV",
                "-t", str(chunk), "-c", "copy", output_path
            ])
            prop = "Normal" if label == "Normal" else "Faulty"
            writer.writerow([i+1 if label == "Normal" else i+51, output_name, prop])
