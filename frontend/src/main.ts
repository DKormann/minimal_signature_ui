const previewContainer = document.getElementById("previewContainer") as HTMLDivElement;
const bubblesContainer = document.getElementById("bubblesContainer") as HTMLDivElement;

const toggleHorizontalFocusButton = document.getElementById("toggleHorizontalFocusButton") as HTMLButtonElement;
const toggleVerticalFocusButton = document.getElementById("toggleVerticalFocusButton") as HTMLButtonElement;
const toggleBubblesButton = document.getElementById("toggleBubblesButton") as HTMLButtonElement;


let overlay : HTMLDivElement | null = null;


let image_blob : Blob | null = null;

const horizontalBandSize = 5.5
const horizontalStepSize = 4.25;
const horizontalFocusTop = 21;

const verticalBandSize = 15.2
const verticalFocusLeft = 3;
const verticalStepSize = 15.3;;


let currentFocusTop = horizontalFocusTop;
let currentFocusLeft = verticalFocusLeft;

let focusMode : 'horizontal' | 'vertical' | 'bubbles' | 'none' = 'none'; 

async function load() {

  let doclist :string[] = []
  await fetch("http://localhost:5000/api/list_documents").then(async (response) => {
    await response.json().then((data) => {doclist = data;})
  });

  let doc_picker = document.querySelector("#fileInput") as HTMLSelectElement

  ((doclist)).forEach((doc) => {
    let option = document.createElement("option");
    option.value = doc;
    option.textContent = doc;
    doc_picker.appendChild(option);

  });

  doc_picker.value = "documents"; // Set default value to "documents"



  doc_picker.addEventListener("change", (event) => {
    let selectedDoc = (event.target as HTMLSelectElement).value;
    console.log("Selected document:", selectedDoc);
  });

  const resetOverlay = (el:HTMLImageElement|undefined)=>{
    bubbleindex = 1;
    currentFocusLeft = verticalFocusLeft;
    currentFocusTop = horizontalFocusTop;

    bubblesContainer.style.display = "none";
    previewContainer.style.display = "block";
    const wrapper = document.createElement("div");
    wrapper.style.position = "relative";
    if (el) {
      wrapper.appendChild(el);
    }
    overlay = document.createElement("div");
    overlay.className = "focus-overlay";

    wrapper.appendChild(overlay);
    previewContainer.appendChild(wrapper);

    const uploader = document.getElementById("uploader") as HTMLDivElement;
    uploader.classList.remove("focused-horizontal", "focused-vertical");

  }

  doc_picker.addEventListener("change", (event) => {

    let selectedDoc = (event.target as HTMLSelectElement).value;
    fetch(`http://localhost:5000/api/get_document/${selectedDoc}`)
      .then(response => {
        if (response.ok) {
          return response.blob();
        } else {
          throw new Error('Network response was not ok');
        }
      })
      .then(blob => {
        image_blob = blob;
        const url = URL.createObjectURL(blob);
        previewContainer.innerHTML = ""; // Clear previous content
        let el;
        if (selectedDoc.endsWith(".pdf")) {
          el = document.createElement("iframe");
          el.src = url;
          el.className = "preview iframe";

        }else if (selectedDoc.endsWith(".jpg") || selectedDoc.endsWith(".png")) {
          el = document.createElement("img");
          el.src = url;
          el.className = "preview";
        } else {
          previewContainer.textContent = "File type not supported.";
          return;
        }

        focusMode = 'none';
        resetOverlay(el as HTMLImageElement);

        toggleFocusMode('bubbles');
        })
  });

  let bubbleindex = 1;

  function showBubbles(){

    bubblesContainer.innerHTML = "";


    if (!image_blob) return
    const imageurl = URL.createObjectURL(image_blob as Blob);

    let mkBubble = (i: number) => {
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubblesContainer.appendChild(bubble);

      bubble.style.backgroundImage = `url(${imageurl})`;
      bubble.style.backgroundSize = "900%";
      bubble.style.backgroundPosition = `${22 + i*(16.3)}% ${currentFocusTop}%`;

      bubble.innerHTML = `<span>${i + 1}</span>`;
      return bubble;
    }

    let namebubble = mkBubble(0);
    namebubble.innerHTML = ""

    namebubble.style.backgroundPosition = `${8}% ${20}%`;
    namebubble.style.marginRight = "50%";

    bubblesContainer.appendChild(namebubble);


    for (let i = 0; i < 5; i++) {
      let bub = mkBubble(i)
      bubblesContainer.appendChild(bub);
      let imgpath = `http://localhost:5000/api/get_snippet/${bubbleindex}/${i}`
      console.log(imgpath);
      
      bub.style.backgroundImage = `url(${imgpath})`;
      bub.style.backgroundSize = "80%";
      bub.style.backgroundPosition = "center";

    }

    
  }

  function toggleFocusMode(mode: 'horizontal' | 'vertical' | 'bubbles') {
    resetOverlay(document.querySelector(".preview") as HTMLImageElement);

    if (focusMode === mode){
      focusMode = 'none';
      return
    }

    focusMode = mode;
    switch (mode){
      case 'horizontal':
        uploader.classList.add("focused-horizontal");
        showHorizontalFocus();
        break;

      case 'vertical':
        uploader.classList.add("focused-vertical");
        showVerticalFocus();
        break;

      case 'bubbles':
        previewContainer.style.display = "none";
        bubblesContainer.style.display = "flex";
        bubblesContainer.innerHTML = "";
        showBubbles();
        uploader.classList.remove("focused-horizontal", "focused-vertical");
        break;
    }
  }

  function showHorizontalFocus() {
    if (!overlay) return;
    overlay.style.setProperty('--h-focus-top', `${currentFocusTop}%`);
    overlay.style.setProperty('--h-focus-bottom', `${currentFocusTop + horizontalBandSize}%`);
  }

  function showVerticalFocus() {
    if (!overlay) return;
    overlay.style.setProperty('--v-focus-left', `${currentFocusLeft}%`);
    overlay.style.setProperty('--v-focus-right', `${currentFocusLeft + verticalBandSize}%`);
  }

  const uploader = document.getElementById("uploader") as HTMLDivElement;
  toggleHorizontalFocusButton.addEventListener("click", () => toggleFocusMode('horizontal'));
  toggleVerticalFocusButton.addEventListener("click", () => toggleFocusMode('vertical'))
  toggleBubblesButton.addEventListener("click", () => toggleFocusMode('bubbles'));

  document.addEventListener('keydown', (event) => {
    

    let moved = false;
    
    if (focusMode === 'horizontal') {
      console.log(event.key);
      
      if (event.key === 'ArrowUp') {
        currentFocusTop = currentFocusTop - horizontalStepSize;
        showHorizontalFocus();
        moved = true;
      } else if (event.key === 'ArrowDown') {
        currentFocusTop = currentFocusTop + horizontalStepSize;
        showHorizontalFocus();
        moved = true;
      }
    } else if (focusMode === 'vertical') {

      if (event.key === 'ArrowLeft') {
        currentFocusLeft = currentFocusLeft - verticalStepSize;
        showVerticalFocus();
        moved = true;
      } else if (event.key === 'ArrowRight') {
        currentFocusLeft = currentFocusLeft + verticalStepSize;
        showVerticalFocus();
        moved = true;
      }
    } else if (focusMode === 'bubbles') {
      if (event.key === 'ArrowLeft') {
        bubbleindex = Math.max(1, bubbleindex - 1);
        showBubbles();
        moved = true;
      } else if (event.key === 'ArrowRight') {
        bubbleindex = Math.min(5, bubbleindex + 1);
        console.log(bubbleindex);
        
        showBubbles();
        moved = true;
      }
    }


    if (moved) {
      event.preventDefault();
    }
  });

  doc_picker.value = "doc2.png"
  doc_picker.dispatchEvent(new Event("change")); // Trigger change event to load the default document
  toggleFocusMode('bubbles'); // Set initial focus mode to bubbles

}

load()




export {}