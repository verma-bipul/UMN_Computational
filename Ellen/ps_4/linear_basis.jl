# This subroutine gives the piecewise linear basis given an input of grid point vectors

function ψ(X::Vector, y::Real, i::Integer)
   if i==1
    return X[i] <= y && y <= X[i+1] ? (X[i+1] -y)/(X[i+1] - X[i]) : 0.0
   elseif i == length(X)
    return X[i-1] <= y && y <= X[i] ? (y - X[i-1])/(X[i] - X[i-1]) : 0.0
   else
    if X[i-1] <= y &&  y <= X[i]
        return (y-X[i-1])/(X[i]-X[i-1])
    elseif X[i] <= y && y <= X[i+1]
        return (X[i+1] - y)/(X[i+1] - X[i])
    else return 0.0
    end
end
end

