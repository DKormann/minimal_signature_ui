const previewContainer = document.getElementById("previewContainer") as HTMLDivElement;

const toggleHorizontalFocusButton = document.getElementById("toggleHorizontalFocusButton") as HTMLButtonElement;
const toggleVerticalFocusButton = document.getElementById("toggleVerticalFocusButton") as HTMLButtonElement;

let overlay : HTMLDivElement | null = null; // Initialize overlay as null


const horizontalBandSize = 4.2
const verticalBandSize = 15.2

const horizontalFocusTop = 23;
const verticalFocusLeft = 3;

let focusMode = 'none'; 


async function load() {

  let doclist :string[] = []
  await fetch("http://localhost:5000/api/list_documents").then(async (response) => {
    
    await response.json().then((data) => {
      console.log(data);

      doclist = data;
    })
  });


  let doc_picker = document.querySelector("#fileInput") as HTMLSelectElement

  doclist.forEach((doc) => {
    let option = document.createElement("option");
    option.value = doc;
    option.textContent = doc;
    doc_picker.appendChild(option);

  });

  doc_picker.addEventListener("change", (event) => {
    let selectedDoc = (event.target as HTMLSelectElement).value;
    console.log("Selected document:", selectedDoc);
  });  

  
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
        
        const wrapper = document.createElement("div");
        wrapper.style.position = "relative";
        wrapper.appendChild(el);
        overlay = document.createElement("div");
        overlay.className = "focus-overlay";

        overlay.style.setProperty('--h-focus-top', `${horizontalFocusTop}%`);
        overlay.style.setProperty('--h-focus-bottom', `${horizontalFocusTop + horizontalBandSize}%`);
        overlay.style.setProperty('--v-focus-left', `${verticalFocusLeft}%`);
        overlay.style.setProperty('--v-focus-right', `${verticalFocusLeft + verticalBandSize}%`); // Assuming focus band size is 10%


        wrapper.appendChild(overlay);
        previewContainer.appendChild(wrapper);

        const uploader = document.getElementById("uploader") as HTMLDivElement;
        uploader.classList.remove("focused-horizontal", "focused-vertical");
        focusMode = 'none';

        })

  });

  const uploader = document.getElementById("uploader") as HTMLDivElement;
  

  toggleHorizontalFocusButton.addEventListener("click", () => {
    uploader.classList.toggle("focused-horizontal");
    uploader.classList.remove("focused-vertical");
    
  });

  toggleVerticalFocusButton.addEventListener("click", () => {
    uploader.classList.toggle("focused-vertical");
    uploader.classList.remove("focused-horizontal");
  });

  document.addEventListener('keydown', (event) => {
    console.log("Key pressed:", event.key);
    console.log("Overlay class list:", document.querySelector(".focus-overlay")?.classList);
    
    
    if (!overlay) return;
    
    let moved = false;
    
    if (uploader.classList.contains("focused-horizontal")) {

      console.log("Horizontal focus mode active");
      
      if (event.key === 'ArrowUp') {
        const currentTop = parseFloat(getComputedStyle(overlay).getPropertyValue('--h-focus-top')) || 35;
        const newTop = Math.max(0, currentTop - horizontalBandSize);
        overlay.style.setProperty('--h-focus-top', `${newTop}%`);
        overlay.style.setProperty('--h-focus-bottom', `${newTop + horizontalBandSize}%`);
        moved = true;
      } else if (event.key === 'ArrowDown') {
        const currentTop = parseFloat(getComputedStyle(overlay).getPropertyValue('--h-focus-top')) || 35;
        const newTop = Math.min(100 - 10, currentTop + horizontalBandSize);
        overlay.style.setProperty('--h-focus-top', `${newTop}%`);
        overlay.style.setProperty('--h-focus-bottom', `${newTop + horizontalBandSize}%`);
        moved = true;
      }
    } else if (uploader.classList.contains("focused-vertical")) {

      if (event.key === 'ArrowLeft') {
        const currentLeft = parseFloat(getComputedStyle(overlay).getPropertyValue('--v-focus-left')) || 35;
        const newLeft = Math.max(0, currentLeft - verticalBandSize);
        overlay.style.setProperty('--v-focus-left', `${newLeft}%`);
        overlay.style.setProperty('--v-focus-right', `${newLeft + verticalBandSize}%`); // Assuming focus band size is 10%
        moved = true;
      } else if (event.key === 'ArrowRight') {
        const currentLeft = parseFloat(getComputedStyle(overlay).getPropertyValue('--v-focus-left')) || 35;
        const newLeft = Math.min(100 - 10, currentLeft + verticalBandSize); // Assuming focus band size is 10%
        overlay.style.setProperty('--v-focus-left', `${newLeft}%`);
        overlay.style.setProperty('--v-focus-right', `${newLeft + verticalBandSize}%`);
        moved = true;
      }
    }
    if (moved) {
      event.preventDefault(); // Prevent default scrolling behavior
      // updateOverlayCssVariables(); // This function is not defined in this context, but you can call it if needed
    }
  });


}

load()


export {}