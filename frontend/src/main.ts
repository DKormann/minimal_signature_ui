
  // <script>
  //   const fileInput = document.getElementById("fileInput");
  //   const previewContainer = document.getElementById("previewContainer");
  //   const uploader = document.getElementById("uploader");
  //   const toggleHorizontalFocusButton = document.getElementById("toggleHorizontalFocusButton");
  //   const toggleVerticalFocusButton = document.getElementById("toggleVerticalFocusButton");

  //   let overlay; 


  //   let focusMode = 'none'; 
  //   const initialHorizontalFocusTop = 35; 
  //   const initialVerticalFocusLeft = 35;  
  //   const focusBandSize = 10;
  //   const focusStep = 1;

  //   let currentHorizontalFocusTop = initialHorizontalFocusTop;
  //   let currentVerticalFocusLeft = initialVerticalFocusLeft;

  //   function updateOverlayCssVariables() {
  //     if (overlay) {
  //       overlay.style.setProperty('--h-focus-top', `${currentHorizontalFocusTop}%`);
  //       overlay.style.setProperty('--h-focus-bottom', `${currentHorizontalFocusTop + focusBandSize}%`);
  //       overlay.style.setProperty('--v-focus-left', `${currentVerticalFocusLeft}%`);
  //       overlay.style.setProperty('--v-focus-right', `${currentVerticalFocusLeft + focusBandSize}%`);
  //     }
  //   }

  //   fileInput.addEventListener("change", () => {
  //     const file = fileInput.files[0];
  //     if (!file) return;

  //     const fileURL = URL.createObjectURL(file);
  //     previewContainer.innerHTML = ""; 

  //     let el;
  //     if (file.type.startsWith("image/")) {
  //       el = document.createElement("img");
  //       el.src = fileURL;
  //       el.className = "preview";
  //     } else if (file.type === "application/pdf") {
  //       el = document.createElement("iframe");
  //       el.src = fileURL;
  //       el.className = "preview iframe";
  //       el.type = "application/pdf";
  //     } else {
  //       previewContainer.textContent = "File type not supported.";
  //       return;
  //     }

  //     const wrapper = document.createElement("div");
  //     wrapper.style.position = "relative";
  //     wrapper.appendChild(el);

  //     overlay = document.createElement("div");
  //     overlay.className = "focus-overlay";
  //     wrapper.appendChild(overlay);

  //     previewContainer.appendChild(wrapper);

  //     // Reset focus state
  //     uploader.classList.remove("focused-horizontal", "focused-vertical");
  //     focusMode = 'none';
  //     currentHorizontalFocusTop = initialHorizontalFocusTop;
  //     currentVerticalFocusLeft = initialVerticalFocusLeft;
  //     updateOverlayCssVariables(); 
  //   });

  //   toggleHorizontalFocusButton.addEventListener("click", () => {
  //     if (focusMode === 'horizontal') {
  //       uploader.classList.remove("focused-horizontal");
  //       focusMode = 'none';
  //     } else {
  //       uploader.classList.remove("focused-vertical"); 
  //       uploader.classList.add("focused-horizontal");
  //       focusMode = 'horizontal';
  //     }
  //   });

  //   toggleVerticalFocusButton.addEventListener("click", () => {
  //     if (focusMode === 'vertical') {
  //       uploader.classList.remove("focused-vertical");
  //       focusMode = 'none';
  //     } else {
  //       uploader.classList.remove("focused-horizontal"); 
  //       uploader.classList.add("focused-vertical");
  //       focusMode = 'vertical';
  //     }
  //   });

  //   document.addEventListener('keydown', (event) => {
  //     if (!overlay || focusMode === 'none') return; 

  //     let moved = false;
  //     if (focusMode === 'horizontal') {
  //       if (event.key === 'ArrowUp') {
  //         currentHorizontalFocusTop = Math.max(0, currentHorizontalFocusTop - focusStep);
  //         moved = true;
  //       } else if (event.key === 'ArrowDown') {
  //         currentHorizontalFocusTop = Math.min(100 - focusBandSize, currentHorizontalFocusTop + focusStep);
  //         moved = true;
  //       }
  //     } else if (focusMode === 'vertical') {
  //       if (event.key === 'ArrowLeft') {
  //         currentVerticalFocusLeft = Math.max(0, currentVerticalFocusLeft - focusStep);
  //         moved = true;
  //       } else if (event.key === 'ArrowRight') {
  //         currentVerticalFocusLeft = Math.min(100 - focusBandSize, currentVerticalFocusLeft + focusStep);
  //         moved = true;
  //       }
  //     }

  //     if (moved) {
  //       event.preventDefault(); 
  //       updateOverlayCssVariables();
  //     }
  //   });
  // </script>


const previewContainer = document.getElementById("previewContainer") as HTMLDivElement;

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
    // console.log("Clicked document:", selectedDoc);
    // if (selectedDoc) {
    //   window.location.href = `http://localhost:5000/api/get_document/${selectedDoc}`;
    // }

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
          // el.type = "application/pdf";
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
        const overlay = document.createElement("div");
        overlay.className = "focus-overlay";
        wrapper.appendChild(overlay);
        previewContainer.appendChild(wrapper);
        // Reset focus state

        // const uploader = document.getElementById("uploader") as HTMLDivElement;
        // uploader.classList.remove("focused-horizontal", "focused-vertical");
        // let focusMode = 'none';
        // const initialHorizontalFocusTop = 35;
        // const initialVerticalFocusLeft = 35;
      })



          

  });
  

}

load()


export {}