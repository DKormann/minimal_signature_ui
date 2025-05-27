




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
  

}

load()


export {}