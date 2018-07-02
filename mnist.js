async function loadMNIST(filename, header) {
    let r = await fetch(`mnist/${filename}`);
    let data = await r.arrayBuffer();
    return new Uint8Array(data).slice(header);
}

function loadAll() {
    let mnist = {};
    loadMNIST('train-images-idx3-ubyte', 16)
        .then(a => {
            mnist.traning_imgs = a;
            return loadMNIST('train-labels-idx1-ubyte', 8);
        }).then(b => {
            mnist.traning_labels = b;
            return loadMNIST('t10k-images-idx3-ubyte', 16);
        }).then(c => {
            mnist.testing_imgs = c;
            return loadMNIST('t10k-labels-idx1-ubyte', 8);
        }).then (d => {
            mnist.testing_labels = d;
        });
    return mnist;
}
