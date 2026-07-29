document.getElementById("checkBtn").addEventListener("click", async () => {
  const res = await fetch("/gpu-check");
  const data = await res.json();
  document.getElementById("gpuResult").textContent = JSON.stringify(data, null, 2);
});
