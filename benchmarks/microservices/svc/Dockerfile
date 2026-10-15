FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod main.go ./
RUN CGO_ENABLED=0 go build -trimpath -o /svc .
FROM alpine:3.20
RUN apk add --no-cache iproute2 curl
COPY --from=build /svc /svc
EXPOSE 8080
ENTRYPOINT ["/svc"]
