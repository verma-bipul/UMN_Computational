## This file contains parameter

## Loading Relevant Packages
using Parameters   # This package will be used to clearly evaluate results for different parameter steady_state

## Parameter set 1
@with_kw struct model_parameters1
    β::Real = 0.95   # discount parameter
    θ::Real = 0.3  # capital share in production

    ρ::Real  = 0.8    # AR(1) coeff
    σ::Real  = 0.5     # variance of random shocks

    ψ::Real  = 0.0  # labor supply coeff
    γ_z::Real = 0.0  # productivity growth parameter     
    γ_n::Real = 0.0  # labor growth parameter
    δ:: Real  = 1  # depreciation
end

## Parameters set 2
@with_kw struct model_parameters2
    β::Real = 0.95   # discount parameter
    θ::Real = 0.3  # capital share in production

    ρ::Real  = 0.8    # AR(1) coeff
    σ::Real  = 0.5     # variance of random shocks

    ψ::Real  = 0.0  # labor supply coeff
    γ_z::Real = 0.0  # productivity growth parameter     
    γ_n::Real = 0.0  # labor growth parameter
    δ:: Real  = 0.05  # depreciation
end

## Parameter set 3

@with_kw struct model_parameters3
    β::Real = 0.95   # discount parameter
    θ::Real = 0.3  # capital share in production

    ρ::Real  = 0.8    # AR(1) coeff
    σ::Real  = 0.5     # variance of random shocks

    ψ::Real  = 0.0  # labor supply coeff
    γ_z::Real = 0.02  # productivity growth parameter     
    γ_n::Real = 0.0  # labor growth parameter
    δ:: Real  = 0.05  # depreciation
end

## Parameters set 4

@with_kw struct model_parameters4
    β::Real = 0.95   # discount parameter
    θ::Real = 0.3  # capital share in production

    ρ::Real  = 0.8    # AR(1) coeff
    σ::Real  = 0.5     # variance of random shocks

    ψ::Real  = 0.0  # labor supply coeff
    γ_z::Real = 0.0  # productivity growth parameter     
    γ_n::Real = 0.02  # labor growth parameter
    δ:: Real  = 0.5  # depreciation
end

## Parameters set 5

@with_kw struct parameters_bipul
    β::Real = 0.95   # discount parameter
    θ::Real = 0.34  # capital share in production

    ρ::Real  = 0.50    # AR(1) coeff
    σ::Real  = 0.05     # variance of random shocks

    ψ::Real  = 1.6  # labor supply coeff
    γ_z::Real = 0.02 # productivity growth parameter     
    γ_n::Real = 0.02  # labor growth parameter
    δ:: Real  = 0.05  # depreciation
end

## Parameters by Jacob

@with_kw struct parameters_jake
    β::Real = 0.965   # discount parameter
    θ::Real = 0.34 # capital share in production

    ρ::Real  = 0.35    # AR(1) coeff
    σ::Real  = 0.5     # variance of random shocks

    ψ::Real  = 1.8  # labor supply coeff
    γ_z::Real = 0.02  # productivity growth parameter     
    γ_n::Real = 0.02  # labor growth parameter
    δ:: Real  = 0.5  # depreciation
end