# Digital Exhibition Space

A 3D digital exhibition space based on Three.js, supporting first-person perspective roaming, 3D model import, and interactive display.

## 🚀 Quick Start

**Simply double-click `index.html` to run the project - no configuration needed!**

### Features:
- ✅ **No server required** - Open directly
- ✅ **No configuration** - Ready to use
- ✅ **Auto-generated textures** - Beautiful program-generated museum-style textures
- ✅ **Full functionality** - All features work normally
- ✅ **No CORS errors** - Works with file:// protocol

### Usage:
1. Find the `index.html` file
2. Double-click to open
3. The browser will automatically run the project
4. If the browser shows a security warning, click "Allow" or "Continue"

## ✨ Features

### Core Features
- **3D Exhibition Space**: Complete virtual exhibition environment with floor, walls, and ceiling
- **Display Case**: Beautiful display case with transparent glass panels, metal frames, and interior lighting
- **GLB Model Import**: Import GLB format 3D models to the scene or display case
- **Model Replacement**: Automatically replaces previous models when importing new ones
- **Real-time Scaling Control**: Dynamically adjust model size (range: 1-500, 100 is baseline)
- **Display Case Pedestal Height Adjustment**: Adjustable pedestal height (range: 5-100)
- **Object Rotation Control**: Manual control of object rotation in display case
- **Rotation Speed Adjustment**: Adjustable rotation speed (range: 1-100)
- **Orbit Camera Interaction**: Hover-to-interact camera control with local wheel zoom

### Interactive Features
- **Camera Control**: Hover inside viewport, drag to orbit, wheel to zoom, right drag to pan, R to reset camera
- **Long-press Support**: Buttons support long-press for continuous adjustment
- **Real-time Preview**: All adjustments are reflected in real-time in the 3D scene
- **Touch Device Support**: Supports touch device interactions

## 🎮 Controls

### Keyboard Controls
- **R** - Reset camera position
- **1** - Pick up / return current object
- **←/→** - Rotate object left/right in display case
- **↑/↓** - Rotate object up/down in display case
- **J/K** - Raise/lower display case pedestal height

### Mouse Controls
- **Hover viewport** - Camera interaction becomes active without extra click
- **Left Drag** - Orbit camera
- **Wheel (inside canvas only)** - Zoom camera and prevent parent-page scrolling
- **Right Drag** - Pan camera
- **Leave viewport** - Parent page wheel scrolling immediately resumes

## 🛠️ Tech Stack

- **Three.js** (v0.144.0): 3D graphics rendering library
- **GLTFLoader**: GLB/GLTF model loader
- **HTML5/CSS3/JavaScript**: Frontend technologies
- **CDN Loading**: Multiple CDN sources as backup

## 📁 Project Structure

```
DigitalExhibition/
├── index.html              # Main HTML file
├── js/
│   └── exhibition.js       # Main program file
├── models/                 # 3D model folder (optional)
│   ├── model.glb          # Example model
│   ├── vase.glb           # Vase model
│   └── README.md          # Model description
├── README.md              # Project description (English)
└── READMECN.md            # Pointer to README (legacy filename)
```

## 📖 Usage

### 1. Import GLB Model to Scene
1. Click the "Import GLB Model to Scene" area in the right control panel
2. Click "Select File" button
3. Select a GLB format 3D model file
4. The model will automatically import to the position in front of the camera
5. Use scale controls to adjust model size (range: 1-500, 100 is baseline)

### 2. Import GLB Model to Display Case
1. Click the "Import GLB Model to Display Case" area in the right control panel
2. Click "Select File" button
3. Select a GLB format 3D model file
4. The model will automatically import into the display case
5. Use scale controls to adjust model size (range: 1-500, 100 is baseline)

### 3. Adjust Display Case Pedestal Height
1. In the "Adjust Display Case Pedestal Height" area
2. Use plus (+) or minus (-) buttons to adjust height
3. Or directly input value in the input box (range: 5-100)
4. Supports long-press for continuous adjustment

### 4. Rotate Object in Display Case
1. In the "Display Case Object Rotation" area
2. Hold "◄ Left Rotate" button for left rotation (counterclockwise)
3. Hold "Right Rotate ►" button for right rotation (clockwise)
4. Release button to stop rotation
5. Use rotation speed control to adjust speed (range: 1-100)

## 📝 Parameter Systems

### Scale System
- **Scene Import**: Scale range 1-500, 100 is baseline
- **Display Case Import**: Scale range 1-500, 100 is baseline
- **Formula**: Actual scale = (scale value / 100) × base scale × 2.5

### Pedestal Height System
- **Range**: 5-100
- **Correspondence**:
  - 5 → 0.5 (original height)
  - 100 → 10.0 (original height)
- **Formula**: Actual height = 0.1 × scale value

### Rotation Speed System
- **Range**: 1-100
- **Correspondence**:
  - 1 → 0.001 (original speed)
  - 100 → 0.1 (original speed)
- **Formula**: Actual speed = scale value × 0.001

## 📌 Notes

1. **Network Connection**: Project depends on CDN to load Three.js and GLTFLoader, requires network connection
2. **Browser Compatibility**: Requires modern browser with WebGL support
3. **Model Format**: Only supports GLB format 3D models
4. **Model Size**: Suggest keeping model files reasonably sized for faster loading
5. **Resource Cleanup**: Automatically cleans up old model resources when importing new models to avoid memory leaks

## 🚀 Deployment

### Local Usage
Simply double-click `index.html` to run locally.

### Deploy to GitHub Pages (Recommended for Online Sharing)

1. **Create GitHub Repository**
   - Log in to [GitHub](https://github.com)
   - Click "+" → "New repository"
   - Repository name: `digital-exhibition` (or any name)
   - Select Public
   - Click "Create repository"

2. **Upload Project Files**
   ```bash
   cd DigitalExhibition
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/digital-exhibition.git
   git branch -M main
   git push -u origin main
   ```

3. **Enable GitHub Pages**
   - In repository page, click "Settings"
   - Find "Pages" in left menu
   - Source: select "main" branch
   - Folder: select "/ (root)"
   - Click "Save"

4. **Access Your Website**
   - Wait 1-2 minutes for GitHub to generate the site
   - Address format: `https://YOUR_USERNAME.github.io/digital-exhibition/`

### Other Deployment Options
- **Netlify**: Drag and drop the folder to deploy
- **Vercel**: Import GitHub repository or upload folder
- **Cloudflare Pages**: Connect GitHub repository

## 🔮 Future Plans

- [ ] Support more 3D model formats (OBJ, FBX, etc.)
- [ ] Add model animation support
- [ ] Add multiple display case support
- [ ] Add model material editing
- [ ] Add scene save/load functionality
- [ ] Add VR support
- [ ] Optimize mobile device experience

## 📄 License

This project uses the MIT License.

## 👨‍💻 Author

Digital Exhibition Space Project

## 🙏 Acknowledgments

- Three.js team for the excellent 3D library
- GLTFLoader developers
- All contributors and users

---

**Enjoy your digital exhibition journey!** 🎨✨
