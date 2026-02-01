const express = require("express");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = process.env.PORT || 3000;

// ================= PATHS =================
const PATHS = {
  FRONTEND: path.join(__dirname, "../frontend"),
  DATA: path.join(__dirname, "../data/processed"),
  IMAGES: path.join(__dirname, "../images"),
  // 👇 هنا بنستخدم الملف الجديد اللي دمجناه
  DB_FILE: path.join(__dirname, "../data/processed/db-movies-final.json"), 
  PYTHON_SCRIPT: path.join(__dirname, "../inference/predict.py"),
};

// ================= DATA LOADING =================
let movieDatabase = [];

try {
  if (fs.existsSync(PATHS.DB_FILE)) {
    const rawData = fs.readFileSync(PATHS.DB_FILE, "utf-8");
    movieDatabase = JSON.parse(rawData);
    console.info(`✅ Database loaded: ${movieDatabase.length} movies ready.`);
  } else {
    console.error(`❌ Error: File not found ${PATHS.DB_FILE}`);
  }
} catch (error) {
  console.error("❌ Critical Error:", error.message);
}

// ================= MIDDLEWARE =================
app.use(express.json());
app.use(express.static(PATHS.FRONTEND));
app.use("/data", express.static(PATHS.DATA));
app.use("/images", express.static(PATHS.IMAGES));

// ================= ROUTES =================

app.get("/", (req, res) => res.sendFile(path.join(PATHS.FRONTEND, "index.html")));

// --- SEARCH API ---
app.post("/api/search", (req, res) => {
  try {
    const { name, genre, year, average_rating } = req.body;
    
    // Normalize Inputs
    const qName = name ? name.toLowerCase().trim() : null;
    const qYear = year ? parseInt(year) : null;
    const qRating = average_rating ? parseFloat(average_rating) : null;

    const filtered = movieDatabase.filter((m) => {
      // 1. Name Match
      // بنستخدم m.movie_name اللي جاي من الـ JSON
      const title = m.movie_name || m.title || "";
      const matchName = qName ? title.toLowerCase().includes(qName) : true;

      // 2. Genre Match
      const matchGenre = genre ? (m.genres || "").includes(genre) : true;

      // 3. Year Match
      const matchYear = qYear ? m.year === qYear : true;

      // 4. Rating Match (دلوقتي عندنا عمود اسمه rating صريح)
      const movieRating = m.rating || 0;
      const matchRating = qRating ? movieRating >= qRating : true;

      return matchName && matchGenre && matchYear && matchRating;
    });

    res.json({ 
      recommendations: filtered.map(m => ({
        title: m.movie_name || m.title,
        year: m.year,
        genres: m.genres,
        rating: m.rating, // الرقم جاهز
        distance: 0
      })) 
    });

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Server Error" });
  }
});

// --- RECOMMEND API ---
app.post("/api/recommend", (req, res) => {
  const pythonProcess = spawn("python", [PATHS.PYTHON_SCRIPT, JSON.stringify(req.body)]);
  let resultBuffer = "";
  
  pythonProcess.stdout.on("data", (data) => resultBuffer += data.toString());
  
  pythonProcess.on("close", () => {
    try {
      res.json(JSON.parse(resultBuffer));
    } catch (e) {
      res.json({ recommendations: [] });
    }
  });
});

app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));