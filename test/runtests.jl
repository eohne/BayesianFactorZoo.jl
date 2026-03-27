using Test
using Random
using Statistics
using LinearAlgebra

using BayesianFactorZoo

@testset "BayesianFactorZoo" begin

    # -----------------------------
    # Deterministic test data
    # -----------------------------
    Random.seed!(42)

    t = 60
    k = 2
    N = 5

    f = randn(t, k)
    R = randn(t, N)

    f1 = f[:, 1:1]
    f2 = f[:, 2:2]

    lambda0 = zeros(k)

    test_res = (;BayFM = [  0.042467117907013484, 0.052076117699321946,-0.2248773224148553],
                 BaySDF_m =  [ 0.03947315822597166, 0.08220556078730791, -0.2059276016252145],
                 BaySDF_r =  [0.01903378549067611],
                 ss_sdf_g =  [ 0.9, 0.755],
                 ss_sdf_l =  [  0.030966516819372444,0.0024465651737632494,-0.0032572200952872875],
                 ss_sdf_v2_g =  [ 0.61, 0.655],
                 dirac_ss = [0.5075]
                 )


    # -----------------------------
    # TwoPassRegression
    # -----------------------------
    @testset "TwoPassRegression" begin
        res = TwoPassRegression(f, R)

        @test hasproperty(res, :lambda)
        @test hasproperty(res, :lambda_gls)
        @test hasproperty(res, :R2_adj)

        @test length(res.lambda) == k+1
        @test length(res.lambda_gls) == k+1
    end

    # -----------------------------
    # SDF_gmm
    # -----------------------------
    @testset "SDF_gmm" begin
        W = Matrix{Float64}(I, N+k, N+k)

        Random.seed!(123)
        res = SDF_gmm(R, f, W)

        @test hasproperty(res, :lambda_gmm)
        @test hasproperty(res, :R2_adj)
        @test isfinite(res.R2_adj)
    end

    # -----------------------------
    # BayesianFM (regression test)
    # -----------------------------
    @testset "BayesianFM" begin
        Random.seed!(123)
        res = BayesianFM(f, R, 200; seed=123)

        @test size(res.lambda_ols_path, 1) == 200
        @test size(res.lambda_gls_path, 1) == 200

        m = vec(mean(res.lambda_ols_path, dims=1))

        @test isapprox(m,
            test_res.BayFM;
            rtol=1e-8, atol=1e-8)
    end

    # -----------------------------
    # BayesianSDF (regression test)
    # -----------------------------
    @testset "BayesianSDF" begin
        Random.seed!(123)
        res = BayesianSDF(f, R, 200; psi0=5.0, d=0.5, seed=123)

        @test size(res.lambda_path, 1) == 200
        @test length(res.R2_path) == 200

        m = vec(mean(res.lambda_path, dims=1))
        r2 = mean(res.R2_path)

        @test isapprox(m,
            test_res.BaySDF_m;
            rtol=1e-8, atol=1e-8)

        @test isapprox(r2,
            first(test_res.BaySDF_r);
            rtol=1e-8, atol=1e-8)
    end

    # -----------------------------
    # continuous_ss_sdf
    # -----------------------------
    @testset "continuous_ss_sdf" begin
        Random.seed!(123)
        res = continuous_ss_sdf(f, R, 200;
            psi0=1.0, r=0.001, aw=1.0, bw=1.0,
            type="OLS", seed=123)

        @test size(res.gamma_path) == (200, k)
        @test size(res.lambda_path, 1) == 200
        @test size(res.sdf_path, 1) == 200

        gamma_mean = vec(mean(res.gamma_path, dims=1))
        lambda_mean = vec(mean(res.lambda_path, dims=1))

        @test isapprox(gamma_mean,
            test_res.ss_sdf_g;
            rtol=1e-8, atol=1e-8)

        @test isapprox(lambda_mean,
            test_res.ss_sdf_l;
            rtol=1e-8, atol=1e-8)
    end

    # -----------------------------
    # continuous_ss_sdf_v2
    # -----------------------------
    @testset "continuous_ss_sdf_v2" begin
        Random.seed!(123)
        res = continuous_ss_sdf_v2(f1, f2, R, 200;
            psi0=1.0, r=0.001, aw=1.0, bw=1.0,
            type="OLS", seed=123)

        @test size(res.gamma_path) == (200, k)
        @test size(res.lambda_path, 1) == 200
        @test size(res.sdf_path, 1) == 200

        gamma_mean = vec(mean(res.gamma_path, dims=1))

        @test isapprox(gamma_mean,
            test_res.ss_sdf_v2_g;
            rtol=1e-8, atol=1e-8)
    end

    # -----------------------------
    # dirac_ss_sdf_pvalue
    # -----------------------------
    @testset "dirac_ss_sdf_pvalue" begin
        Random.seed!(123)
        res = dirac_ss_sdf_pvalue(f, R, 200, lambda0; seed=123)

        @test size(res.gamma_path) == (200, k)
        @test size(res.lambda_path, 1) == 200

        gamma_mean = mean(res.gamma_path)

        @test isapprox(gamma_mean,
            first(test_res.dirac_ss);
            rtol=1e-8, atol=1e-8)
    end

end