// deno-lint-ignore-file no-empty-interface no-var
// Copyright 2018-2024 the Deno authors. All rights reserved. MIT license.

/// <reference no-default-lib="true" />
/// <reference lib="esnext" />

declare enum MLDeviceType {
    "cpu",
    "gpu",
    "npu",
}

declare enum MLPowerPreference {
    "default",
    "high-performance",
    "low-power",
}

declare interface MLContextOptions {
    deviceType: MLDeviceType;
    powerPreference: MLPowerPreference;
}

declare interface ML {
    createContext(options?: MLContextOptions): Promise<MLContext>;
    createContext(gpuDevice: GPUDevice): Promise<MLContext>;
}

declare type MLNamedArrayBufferViews = Record<string, ArrayBufferView>;

declare interface MLComputeResult {
    inputs: MLNamedArrayBufferViews;
    outputs: MLNamedArrayBufferViews;
}

declare interface MLContext {
    compute(
        graph: MLGraph,
        inputs: MLNamedArrayBufferViews,
        outputs: MLNamedArrayBufferViews,
    ): Promise<MLComputeResult>;
}

declare interface MLGraph {}

declare enum MLInputOperandLayout {
    "nchw",
    "nhwc",
}

declare enum MLOperandDataType {
    "float32",
    "float16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "int8",
    "uint8",
}

declare interface MLOperandDescriptor {
    dataType: MLOperandDataType;
    dimensions: number[];
}

declare interface MLOperand {
    dataType(): MLOperandDataType;
    shape(): number[];
}

declare interface MLActivation {}

declare type MLNamedOperands = Record<string, MLOperand>;

declare interface MLGraphBuilder {
    input(name: string, descriptor: MLOperandDescriptor): MLOperand;

    constant(
        descriptor: MLOperandDescriptor,
        bufferView: ArrayBufferView,
    ): MLOperand;

    constant(type: MLOperandDataType, value: number): MLOperand;

    build(outputs: MLNamedOperands): Promise<MLGraph>;

    argMin(input: MLOperand, options?: MLArgMinMaxOptions): MLOperand;

    argMax(input: MLOperand, options?: MLArgMinMaxOptions): MLOperand;

    batchNormalization(
        input: MLOperand,
        mean: MLOperand,
        variance: MLOperand,
        options?: MLBatchNormalizationOptions,
    ): MLOperand;

    cast(input: MLOperand, type: MLOperandDataType): MLOperand;

    clamp(input: MLOperand, options?: MLClampOptions): MLOperand;

    concat(inputs: MLOperand[], axis: number): MLOperand;

    conv2d(
        input: MLOperand,
        filter: MLOperand,
        options?: MLConv2dOptions,
    ): MLOperand;

    convTranspose2d(
        input: MLOperand,
        filter: MLOperand,
        options?: MLConvTranspose2dOptions,
    ): MLOperand;

    add(a: MLOperand, b: MLOperand): MLOperand;

    sub(a: MLOperand, b: MLOperand): MLOperand;

    mul(a: MLOperand, b: MLOperand): MLOperand;

    div(a: MLOperand, b: MLOperand): MLOperand;

    max(a: MLOperand, b: MLOperand): MLOperand;

    min(a: MLOperand, b: MLOperand): MLOperand;

    pow(a: MLOperand, b: MLOperand): MLOperand;

    equal(a: MLOperand, b: MLOperand): MLOperand;

    greater(a: MLOperand, b: MLOperand): MLOperand;

    greaterOrEqual(a: MLOperand, b: MLOperand): MLOperand;

    lesser(a: MLOperand, b: MLOperand): MLOperand;

    lesserOrEqual(a: MLOperand, b: MLOperand): MLOperand;

    logicalNot(a: MLOperand): MLOperand;

    abs(input: MLOperand): MLOperand;

    ceil(input: MLOperand): MLOperand;

    cos(input: MLOperand): MLOperand;

    erf(input: MLOperand): MLOperand;

    exp(input: MLOperand): MLOperand;

    floor(input: MLOperand): MLOperand;

    identity(input: MLOperand): MLOperand;

    log(input: MLOperand): MLOperand;

    neg(input: MLOperand): MLOperand;

    reciprocal(input: MLOperand): MLOperand;

    sin(input: MLOperand): MLOperand;

    sqrt(input: MLOperand): MLOperand;

    tan(input: MLOperand): MLOperand;

    elu(input: MLOperand, options?: MLEluOptions): MLOperand;

    elu(options?: MLEluOptions): MLActivation;

    expand(input: MLOperand, newShape: number[]): MLOperand;

    gather(
        input: MLOperand,
        indices: MLOperand,
        options?: MLGatherOptions,
    ): MLOperand;

    gelu(input: MLOperand): MLOperand;

    gelu(): MLActivation;

    gemm(a: MLOperand, b: MLOperand, options?: MLGemmOptions): MLOperand;

    gru(
        input: MLOperand,
        weight: MLOperand,
        recurrentWeight: MLOperand,
        steps: number,
        hiddenSize: number,
        options?: MLGruOptions,
    ): MLOperand[];

    gruCell(
        input: MLOperand,
        weight: MLOperand,
        recurrentWeight: MLOperand,
        hiddenState: MLOperand,
        hiddenSize: number,
        options?: MLGruCellOptions,
    ): MLOperand;

    hardSigmoid(input: MLOperand, options?: MLHardSigmoidOptions): MLOperand;

    hardSigmoid(options?: MLHardSigmoidOptions): MLActivation;

    hardSwish(input: MLOperand): MLOperand;

    hardSwish(): MLActivation;

    instanceNormalization(
        input: MLOperand,
        options?: MLInstanceNormalizationOptions,
    ): MLOperand;

    layerNormalization(
        input: MLOperand,
        options?: MLLayerNormalizationOptions,
    ): MLOperand;

    leakyRelu(input: MLOperand, options?: MLLeakyReluOptions): MLOperand;

    leakyRelu(options?: MLLeakyReluOptions): MLActivation;

    linear(input: MLOperand, options?: MLLinearOptions): MLOperand;

    linear(options?: MLLinearOptions): MLActivation;

    lstm(
        input: MLOperand,
        weight: MLOperand,
        recurrentWeight: MLOperand,
        steps: number,
        hiddenSize: number,
        options?: MLLstmOptions,
    ): MLOperand[];

    lstmCell(
        input: MLOperand,
        weight: MLOperand,
        recurrentWeight: MLOperand,
        hiddenState: MLOperand,
        cellState: MLOperand,
        hiddenSize: number,
        options?: MLLstmCellOptions,
    ): MLOperand[];

    matmul(a: MLOperand, b: MLOperand): MLOperand;

    pad(
        input: MLOperand,
        beginningPadding: number[],
        endingPadding: number[],
        options?: MLPadOptions,
    ): MLOperand;

    averagePool2d(
        input: MLOperand,
        options?: MLPool2dOptions,
    ): MLOperand;

    l2Pool2d(input: MLOperand, options?: MLPool2dOptions): MLOperand;

    maxPool2d(input: MLOperand, options?: MLPool2dOptions): MLOperand;

    prelu(input: MLOperand, slope: MLOperand): MLOperand;

    reduceL1(input: MLOperand, options?: MLReduceOptions): MLOperand;

    reduceL2(input: MLOperand, options?: MLReduceOptions): MLOperand;

    reduceLogSum(input: MLOperand, options?: MLReduceOptions): MLOperand;

    reduceLogSumExp(input: MLOperand, options?: MLReduceOptions): MLOperand;

    reduceMax(input: MLOperand, options?: MLReduceOptions): MLOperand;

    reduceMean(input: MLOperand, options?: MLReduceOptions): MLOperand;

    reduceMin(input: MLOperand, options?: MLReduceOptions): MLOperand;

    reduceProduct(input: MLOperand, options?: MLReduceOptions): MLOperand;

    reduceSum(input: MLOperand, options?: MLReduceOptions): MLOperand;

    reduceSumSquare(input: MLOperand, options?: MLReduceOptions): MLOperand;

    relu(input: MLOperand): MLOperand;

    relu(): MLActivation;

    // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-resample2d-method
}

declare var MLGraphBuilder: {
    prototype: MLGraphBuilder;
    new (context: MLContext): MLGraphBuilder;
};

declare interface MLArgMinMaxOptions {
    axes: number[];
    keepDimensions: boolean;
    selectLastIndex: boolean;
}

declare interface MLBatchNormalizationOptions {
    scale: MLOperand;
    bias: MLOperand;
    axis: number;
    epsilon: number;
}

declare interface MLClampOptions {
    minValue: number;
    maxValue: number;
}

declare enum MLConv2dFilterOperandLayout {
    "oihw",
    "hwio",
    "ohwi",
    "ihwo",
}

declare interface MLConv2dOptions {
    padding: number[];
    strides: number[];
    dilations: number[];
    groups: number;
    inputLayout: MLInputOperandLayout;
    filterLayout: MLConv2dFilterOperandLayout;
    bias: MLOperand;
}

declare enum MLConvTranspose2dFilterOperandLayout {
    "iohw",
    "hwoi",
    "ohwi",
}

declare interface MLConvTranspose2dOptions {
    padding: number[];
    strides: number[];
    dilations: number[];
    outputPadding: number[];
    outputSizes: number[];
    groups: number;
    inputLayout: MLInputOperandLayout;
    filterLayout: MLConvTranspose2dFilterOperandLayout;
    bias: MLOperand;
}

declare interface MLEluOptions {
    alpha: number;
}

declare interface MLGatherOptions {
    axis: number;
}

declare interface MLGemmOptions {
    c: MLOperand;
    alpha: number;
    beta: number;
    aTranspose: boolean;
    bTranspose: boolean;
}

declare enum MLGruWeightLayout {
    "zrn",
    "rzn",
}

declare enum MLRecurrentNetworkDirection {
    "forward",
    "backward",
    "both",
}

declare interface MLGruOptions {
    bias: MLOperand;
    recurrentBias: MLOperand;
    initialHiddenState: MLOperand;
    resetAfter: boolean;
    returnSequence: boolean;
    direction: MLRecurrentNetworkDirection;
    layout: MLGruWeightLayout;
    activations: MLActivation[];
}

declare interface MLGruCellOptions {
    bias: MLOperand;
    recurrentBias: MLOperand;
    resetAfter: boolean;
    layout: MLGruWeightLayout;
    activations: MLActivation[];
}

declare interface MLHardSigmoidOptions {
    alpha: number;
    beta: number;
}

declare interface MLInstanceNormalizationOptions {
    scale: MLOperand;
    bias: MLOperand;
    epsilon: number;
    layout: MLInputOperandLayout;
}

declare interface MLLayerNormalizationOptions {
    scale: MLOperand;
    bias: MLOperand;
    axes: number[];
    epsilon: number;
}

declare interface MLLeakyReluOptions {
    alpha: number;
}

declare interface MLLinearOptions {
    alpha: number;
    beta: number;
}

declare enum MLLstmWeightLayout {
    "iofg",
    "ifgo",
}

declare interface MLLstmOptions {
    bias: MLOperand;
    recurrentBias: MLOperand;
    peepholeWeight: MLOperand;
    initialHiddenState: MLOperand;
    initialCellState: MLOperand;
    returnSequence: boolean;
    direction: MLRecurrentNetworkDirection;
    layout: MLLstmWeightLayout;
    activations: MLActivation[];
}

declare interface MLLstmCellOptions {
    bias: MLOperand;
    recurrentBias: MLOperand;
    peepholeWeight: MLOperand;
    layout: MLLstmWeightLayout;
    activations: MLActivation[];
}

declare enum MLPaddingMode {
    "constant",
    "edge",
    "reflection",
    "symmetric",
}

declare interface MLPadOptions {
    mode: MLPaddingMode;
    value: number;
}

declare enum MLRoundingType {
    "floor",
    "ceil",
}

declare interface MLPool2dOptions {
    windowDimensions: number[];
    padding: number[];
    strides: number[];
    dilations: number[];
    layout: MLInputOperandLayout;
    roundingType: MLRoundingType;
    outputSizes: number[];
}

declare interface MLReduceOptions {
    axes: number[];
    keepDimensions: boolean;
}