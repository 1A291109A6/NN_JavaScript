var info = [];
var input = [];
var output = [];

var x = [];
var z = [];
var weights = [];

//calculate the dot product
//A is a 1*n matrix
//B is a n*len matrix
function dot(A, B, len) {
    let C = [];
    for (let i = 0; i < len; i++) {
        let sum = 0;
        for (let j = 0; j < A.length; j++) {
            sum += A[j] * B[A.length * i + j];
        }
        C.push(sum);
    }
    return C;
}

function sum_matrix(A, B) {
    let C = [];
    for (let i = 0; i < A.length; i++) {
        C.push(A[i] + B[i]);
    }
    return C;
}

function weights_extraction(start, len) {
    let C = [];
    for (let i = start; i < start + len; i++) {
        C.push(weights[i]);
    }
    return C;
}

function rnorm(sd) {
    return sd * Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random())
}

function activation(type, A) {
    let result = [];
    switch (type) {
        case "sigmoid":
            for (let i = 0; i < A.length; i++) {
                result.push(1 / (1 + Math.exp(-1 * A[i])));
            }
            return result;
        case "liner":
            return A;
        case "ReLU":
            for (let i = 0; i < A.length; i++) {
                if (A[i] > 0) {
                    result.push(A[i]);
                } else {
                    result.push(0);
                }
            }
            return result;
    }
}

function Initialize(type) {
    weights = [];
    for (let i = 0; i < info[0] * info[1]; i++) {
        switch (type) {
            case 1:
                weights.push(rnorm(1 / Math.sqrt(info[0])));
                break;
            case 2:
                weights.push(rnorm(Math.sqrt(2 / info[0])));
                break;
        }
    }
    let bias = new Array(info[1]).fill(0);
    weights = weights.concat(bias);
    for (i = 0; i < (info.length - 3) / 2; i++) {
        let layer = i * 2 + 1;
        for (let j = 0; j < info[layer] * info[layer + 2]; j++) {
            switch (type) {
                case 1:
                    weights.push(rnorm(1 / Math.sqrt(info[0])));
                    break;
                case 2:
                    weights.push(rnorm(Math.sqrt(2 / info[0])));
                    break;
            }
        }
        bias = new Array(info[layer + 2]).fill(0);
        weights = weights.concat(bias);
    }
}

function Forward() {
    let keep1 = [];
    let result = [];
    x = input;
    num = info[0] * info[1];
    result = sum_matrix(dot(x, weights_extraction(0, num), info[1]), weights_extraction(num, info[1]));
    z = z.concat(result);
    keep1 = activation(info[2], result);
    x = x.concat(keep1);
    num += info[2];
    for (i = 0; i < (info.length - 3) / 2; i++) {
        let layer = i * 2 + 1;
        result = sum_matrix(dot(keep1, weights_extraction(num, info[layer] * info[layer + 2]), info[layer + 2]), weights_extraction(num + info[layer] * info[layer + 2], info[layer + 2]));
        z = z.concat(result);
        keep1 = activation(info[layer + 3], result);
        x = x.concat(keep1);
        num += info[layer + 2] * (1 + info[layer]);
    }
    output = keep1;
}